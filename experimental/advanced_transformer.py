'''
This training script is based on training TensorFlow tutorial which can be found in:
https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb

Copyright 2022 The TensorFlow Authors.
https://www.apache.org/licenses/LICENSE-2.0
'''

import numpy as np
from positional_encodings.tf_encodings import TFPositionalEncoding1D, TFSummer
import tensorflow as tf
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-2-vocab.txt')
N_CLASSES = tokenizer.vocab_size

HIST_LEN = 20
DROPOUT = 0.2
LEARNING_RATE = 0.0001

SPEAKER_SIZE = 256
SECOND_SIZE = 256
OTHER_SIZE = 256
OVERALL_SIZE = N_CLASSES

NUM_LAYERS = 6
INPUT_SIZE = 128
OUTPUT_SIZE = 128
DFF = 2048
NUM_ATTENTION_HEADS = 8


def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=DFF, depth=d_model)
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


def point_wise_feed_forward_network(
    d_model,
    dff
    ):
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*,
                   d_model,
                   num_attention_heads,
                   dff,
                   dropout_rate=0.1
                   ):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            training=training,
            )

        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout1(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 dff,
                 input_vocab_size,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
          x = self.dropout(x, training=training)

          for i in range(self.num_layers):
              x = self.enc_layers[i](x, training)

          return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_attention_heads,
                 dff,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.mha_masked = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.mha_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, mask, enc_output, training):
        self_attention_mask = None
        if mask is not None:
          mask1 = mask[:, :, None]
          mask2 = mask[:, None, :]
          self_attention_mask = mask1 & mask2

        attn_masked, _ = self.mha_masked(
            query=x,
            value=x,
            key=x,
            attention_mask=self_attention_mask,
            use_causal_mask=True, 
            return_attention_scores=True,
            training=training 
            )

        out1 = self.layernorm1(attn_masked + x)

        attn_cross, _ = self.mha_cross(
            query=out1,
            value=enc_output,
            key=enc_output,
            return_attention_scores=True,
            training=training 
        )

        out2 = self.layernorm2(attn_cross + out1)

        ffn_output = self.ffn(out2) 
        ffn_output = self.dropout1(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 dff,
                 target_vocab_size,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
      
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(target_vocab_size, d_model)

        self.dec_layers = [
            DecoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training):
        mask = self.pos_embedding.compute_mask(x)
        x = self.pos_embedding(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, mask, enc_output, training)
        return x


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(tf.cast(real, dtype=tf.dtypes.int64), tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def create_model():
    # INPUTS
    input_target = tf.keras.layers.Input(shape=(OUTPUT_SIZE))

    # SPEAKER
    input_speaker = tf.keras.layers.Input(shape=(HIST_LEN, SPEAKER_SIZE))
    embedding_speaker = TFSummer(TFPositionalEncoding1D(SPEAKER_SIZE))(input_speaker)
    lstm_speaker = tf.keras.layers.LSTM(128, return_sequences=False)(embedding_speaker)
    bnorm_speaker = tf.keras.layers.BatchNormalization()(lstm_speaker)
    relu_speaker = tf.keras.layers.PReLU()(bnorm_speaker)
    dropout_speaker = tf.keras.layers.Dropout(0.05)(relu_speaker)
    model_speaker = tf.keras.models.Model(inputs=input_speaker, outputs=dropout_speaker)

    # SECOND
    input_second = tf.keras.layers.Input(shape=(HIST_LEN, SECOND_SIZE))
    embedding_second = TFSummer(TFPositionalEncoding1D(SECOND_SIZE))(input_second)
    lstm_second = tf.keras.layers.LSTM(128, return_sequences=False)(embedding_second)
    bnorm_second = tf.keras.layers.BatchNormalization()(lstm_second)
    relu_second = tf.keras.layers.PReLU()(bnorm_second)
    dropout_second = tf.keras.layers.Dropout(0.05)(relu_second)
    model_second = tf.keras.models.Model(inputs=input_second, outputs=dropout_second)

    # OTHER
    input_other = tf.keras.layers.Input(shape=(HIST_LEN, OTHER_SIZE))
    embedding_other = TFSummer(TFPositionalEncoding1D(OTHER_SIZE))(input_other)
    lstm_other = tf.keras.layers.LSTM(64, return_sequences=False)(embedding_other)
    bnorm_other = tf.keras.layers.BatchNormalization()(lstm_other)
    relu_other = tf.keras.layers.PReLU()(bnorm_other)
    dropout_other = tf.keras.layers.Dropout(0.05)(relu_other)
    model_other = tf.keras.models.Model(inputs=input_other, outputs=dropout_other)

    # OVERALL
    input_overall = tf.keras.layers.Input(shape=(OVERALL_SIZE))
    dense_overall = tf.keras.layers.Dense(1024)(input_overall)
    bnorm_overall = tf.keras.layers.BatchNormalization()(dense_overall)
    relu_overall = tf.keras.layers.PReLU()(bnorm_overall)
    dropout_overall = tf.keras.layers.Dropout(0.05)(relu_overall)
    model_overall = tf.keras.models.Model(inputs=input_overall, outputs=dropout_overall)

    # COMBINED
    input_time = tf.keras.layers.Input(shape=(1))
    combined = tf.keras.layers.Concatenate()([model_speaker.output,
                                              model_second.output,
                                              model_other.output,
                                              input_time,
                                              model_overall.output])
    dense = tf.keras.layers.Dense(INPUT_SIZE*INPUT_SIZE)(combined)
    bnorm = tf.keras.layers.BatchNormalization()(dense)
    relu = tf.keras.layers.PReLU()(bnorm)
    dropout = tf.keras.layers.Dropout(0.1)(relu)
    reshape = tf.keras.layers.Reshape((INPUT_SIZE, INPUT_SIZE))(dropout)
    
    # Transformer
    encoder = Encoder(
        num_layers=NUM_LAYERS,
        d_model=INPUT_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dff=DFF,
        input_vocab_size=N_CLASSES,
        dropout_rate=0.1)(reshape)
    decoder = Decoder(
        num_layers=NUM_LAYERS,
        d_model=OUTPUT_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dff=DFF,
        target_vocab_size=N_CLASSES,
        dropout_rate=0.1)(input_target, encoder, None)
    final_layer = tf.keras.layers.Dense(N_CLASSES)(decoder)
    model = tf.keras.models.Model(inputs=[
        model_speaker.input,
        model_second.input,
        model_other.input,
        input_time,
        input_overall,
        input_target], outputs=final_layer)
    
    opt = tf.keras.optimizers.Adam(LEARNING_RATE,
               beta_1=0.9,
               beta_2=0.98,
               epsilon=1e-9)
    
    model.compile(opt, loss_function, metrics=[accuracy_function])
    model.summary()
    return model
