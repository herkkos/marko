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

HIST_LEN = 1
LEARNING_RATE = 0.0001

SPEAKER_SIZE = 128
OVERALL_SIZE = 9999

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

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha3 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha4 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha5 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha6 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha7 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha8 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha9 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )
        self.mha10 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            )

        self.concat = tf.keras.layers.Concatenate(axis=-2)

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model*5,
            dropout=dropout_rate,
            )
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm6 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm7 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm8 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm9 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm10 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm11 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        # self.reshape = tf.keras.layers.Reshape((640, 640))

    def call(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, training):
        attn_output1 = self.mha1(query=x1, value=x1, key=x1, training=training)
        attn_output2 = self.mha2(query=x2, value=x2, key=x2, training=training)
        attn_output3 = self.mha3(query=x3, value=x3, key=x3, training=training)
        attn_output4 = self.mha4(query=x4, value=x4, key=x4, training=training)
        attn_output5 = self.mha5(query=x5, value=x5, key=x5, training=training)
        attn_output6 = self.mha6(query=x6, value=x6, key=x6, training=training)
        attn_output7 = self.mha7(query=x7, value=x7, key=x7, training=training)
        attn_output8 = self.mha8(query=x8, value=x8, key=x8, training=training)
        attn_output9 = self.mha9(query=x9, value=x9, key=x9, training=training)
        attn_output10 = self.mha10(query=x10, value=x10, key=x10, training=training)

        out1 = self.layernorm1(x1 + attn_output1)
        out2 = self.layernorm1(x2 + attn_output2)
        out3 = self.layernorm1(x3 + attn_output3)
        out4 = self.layernorm1(x4 + attn_output4)
        out5 = self.layernorm1(x5 + attn_output5)
        out6 = self.layernorm1(x6 + attn_output6)
        out7 = self.layernorm1(x7 + attn_output7)
        out8 = self.layernorm1(x8 + attn_output8)
        out9 = self.layernorm1(x9 + attn_output9)
        out10 = self.layernorm1(x10 + attn_output10)

        concat_output = self.concat([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10])

        attn_output = self.mha(query=concat_output, value=concat_output, key=concat_output, training=training)

        out11 = self.layernorm11(concat_output + attn_output)

        ffn_output = self.ffn(out11)
        ffn_output = self.dropout1(ffn_output, training=training)
        out = self.layernorm(out11 + ffn_output)

        # out = self.reshape(out)

        return out

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

        self.pos_embedding = TFSummer(TFPositionalEncoding1D(SPEAKER_SIZE))

        self.enc_layers = [
            EncoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout5 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout6 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout7 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout8 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout9 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout10 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, training):
        x1 = self.pos_embedding(x1)
        x2 = self.pos_embedding(x2)
        x3 = self.pos_embedding(x3)
        x4 = self.pos_embedding(x4)
        x5 = self.pos_embedding(x5)
        x6 = self.pos_embedding(x6)
        x7 = self.pos_embedding(x7)
        x8 = self.pos_embedding(x8)
        x9 = self.pos_embedding(x9)
        x10 = self.pos_embedding(x10)
        
        x1 = self.dropout1(x1, training=training)
        x2 = self.dropout2(x2, training=training)
        x3 = self.dropout3(x3, training=training)
        x4 = self.dropout4(x4, training=training)
        x5 = self.dropout5(x5, training=training)
        x6 = self.dropout6(x6, training=training)
        x7 = self.dropout7(x7, training=training)
        x8 = self.dropout8(x8, training=training)
        x9 = self.dropout9(x9, training=training)
        x10 = self.dropout10(x10, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, training)

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

        self.expand = tf.keras.layers.Reshape((1, 128, 128))

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

        # self_attention_mask = self.expand(self_attention_mask)
        # x = self.expand(x)

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

    input_1 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_2 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_3 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_4 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_5 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_6 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_7 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_8 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_9 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
    input_10 = tf.keras.layers.Input(shape=(1, SPEAKER_SIZE))
 
    # Transformer
    encoder = Encoder(
        num_layers=NUM_LAYERS,
        d_model=INPUT_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dff=DFF,
        input_vocab_size=N_CLASSES,
        dropout_rate=0.1)(input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10)
    decoder = Decoder(
        num_layers=NUM_LAYERS,
        d_model=OUTPUT_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dff=DFF,
        target_vocab_size=N_CLASSES,
        dropout_rate=0.1)(input_target, encoder, None)
    final_layer = tf.keras.layers.Dense(N_CLASSES)(decoder)
    model = tf.keras.models.Model(inputs=[
        input_1,
        input_2,
        input_3,
        input_4,
        input_5,
        input_6,
        input_7,
        input_8,
        input_9,
        input_10,
        input_target], outputs=final_layer)
    
    opt = tf.keras.optimizers.Adam(LEARNING_RATE,
               beta_1=0.9,
               beta_2=0.98,
               epsilon=1e-9)
    
    model.compile(opt, loss_function, metrics=[accuracy_function])
    model.summary()
    return model
