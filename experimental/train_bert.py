import argparse
import os
import string
import time

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-marko-vocab.txt')
N_CLASSES = tokenizer.vocab_size

MODEL_DIR = 'bert_kaikki_1'
checkpoint_path = './checkpoints_kaikki_5/train'
CATEGORY_FILE = '../categories_bert.txt'

HIST_LEN = 2
DROPOUT = 0.1
EPOCHS = 1000

num_layers = 6
INPUT_SIZE = 128
OUTPUT_SIZE = 128
DFF = 2048
num_attention_heads = 8


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
    
    def call(self, x, training, mask):
        if mask is not None:
          mask1 = mask[:, :, None]
          mask2 = mask[:, None, :]
          attention_mask = mask1 & mask2
        else:
          attention_mask = None
      
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=attention_mask,
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
                 dropout_rate=0.1
                 ):
        super().__init__()
      
        self.d_model = d_model
        self.num_layers = num_layers
      
        self.pos_embedding = PositionalEmbedding(input_vocab_size, d_model)
      
        self.enc_layers = [
            EncoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def compute_mask(self, x, previous_mask=None):
        return self.pos_embedding.compute_mask(x, previous_mask)
    
    def call(self, x, training):
          mask = self.compute_mask(x)
          x = self.pos_embedding(x)
          x = self.dropout(x, training=training)
        
          for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
          return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_attention_heads,
                 dff,
                 dropout_rate=0.1
                 ):
        super().__init__()
      
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
    
    def call(self, x, mask, enc_output, enc_mask, training):
        self_attention_mask = None
        if mask is not None:
          mask1 = mask[:, :, None]
          mask2 = mask[:, None, :]
          self_attention_mask = mask1 & mask2
      
        attn_masked, attn_weights_masked = self.mha_masked(
            query=x,
            value=x,
            key=x,
            attention_mask=self_attention_mask,
            use_causal_mask=True, 
            return_attention_scores=True,
            training=training 
            )
      
        out1 = self.layernorm1(attn_masked + x)
      
        attention_mask = None
        if mask is not None and enc_mask is not None:
          mask1 = mask[:, :, None]
          mask2 = enc_mask[:, None, :]
          attention_mask = mask1 & mask2
      
        attn_cross, attn_weights_cross = self.mha_cross(
            query=out1,
            value=enc_output,
            key=enc_output,
            attention_mask=attention_mask,
            return_attention_scores=True,
            training=training 
        )
      
        out2 = self.layernorm2(attn_cross + out1)
      
        ffn_output = self.ffn(out2) 
        ffn_output = self.dropout1(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
      
        return out3, attn_weights_masked, attn_weights_cross

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 d_model,
                 num_attention_heads,
                 dff,
                 target_vocab_size,
                 dropout_rate=0.1
                 ):
        super().__init__()
      
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
    
    def call(self, x, enc_output, enc_mask, training):
        attention_weights = {}
      
        mask = self.pos_embedding.compute_mask(x)
        x = self.pos_embedding(x)
      
        x = self.dropout(x, training=training)
      
        for i in range(self.num_layers):
          x, block1, block2  = self.dec_layers[i](x, mask, enc_output, enc_mask, training)
          attention_weights[f'decoder_layer{i+1}_block1'] = block1
          attention_weights[f'decoder_layer{i+1}_block2'] = block2
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self,
                 *,
                 num_layers,
                 input_size,
                 output_size,
                 num_attention_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 dropout_rate=0.1
                 ):
        super().__init__()
        
        self.encoder = Encoder(
          num_layers=num_layers,
          d_model=input_size,
          num_attention_heads=num_attention_heads,
          dff=dff,
          input_vocab_size=input_vocab_size,
          dropout_rate=dropout_rate
          )
      
        self.decoder = Decoder(
          num_layers=num_layers,
          d_model=output_size,
          num_attention_heads=num_attention_heads,
          dff=dff,
          target_vocab_size=target_vocab_size,
          dropout_rate=dropout_rate
          )
      
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs, training):
        inp, tar = inputs
        enc_output = self.encoder(inp, training)
        enc_mask = self.encoder.compute_mask(inp)
        
        dec_output, attention_weights = self.decoder(
            tar, enc_output, enc_mask, training)
      
        final_output = self.final_layer(dec_output)      
        return final_output, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
      
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
      
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
      
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class Translator(tf.Module):
    def __init__(self, tokenizer, transformer, input_size=128, output_size=32):
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.input_size = input_size
        self.output_size = output_size
    
    def __call__(self, inputs, max_length=128):
        encoder_input = []
        for sentence in inputs:
            encoder_input += self.tokenizer(sentence)['input_ids'][1:-1]
      
        if len(encoder_input) > self.input_size:
            encoder_input = encoder_input[len(encoder_input)-self.input_size:]
        else:
            encoder_input = encoder_input + [tokenizer.pad_token_id] * self.input_size
            encoder_input = encoder_input[:self.input_size]

        encoder_input = tf.constant(encoder_input, dtype=tf.int64)[tf.newaxis]

        start_end = self.tokenizer([''])['input_ids'][0]
        start = tf.constant(start_end[0], dtype=tf.int64)[tf.newaxis]
        end = tf.constant(start_end[1], dtype=tf.int64)[tf.newaxis]
                
        output_array = tf.TensorArray(dtype=tf.int64, size=max_length, dynamic_size=True)
        output_array = output_array.write(0, start)
      
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, i:i+1, :]
            
            pp = predictions.numpy().squeeze()
            for _ in range(100):
                pp -= pp.mean()
                pp = pp**5
                pp[pp < 0] = 0
                pp = np.power(pp, 1/5)
                if np.count_nonzero(pp) < 20:
                    break
            pp = pp**5
            pp /= pp.sum()
            predicted_id = np.random.choice(np.arange(self.tokenizer.vocab_size), p=pp)
            output_array = output_array.write(i+1, np.expand_dims(predicted_id, 0))
            if predicted_id == end.numpy()[0]:
                break
            
        output = tf.squeeze(tf.transpose(output_array.stack()))
        text = self.tokenizer.decode(list(output), skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(output)
        return text, tokens


def train_split(args):
    transformer = Transformer(
        num_layers=num_layers,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_attention_heads=num_attention_heads,
        dff=DFF,
        input_vocab_size=N_CLASSES,
        target_vocab_size=N_CLASSES,
        dropout_rate=DROPOUT)

    learning_rate = CustomSchedule(INPUT_SIZE, 5000)
    optimizer = Adam(learning_rate,
               beta_1=0.9,
               beta_2=0.98,
               epsilon=1e-9)
    
    loss_object = SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none')
    
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
    
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
    
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
    
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print('Latest checkpoint restored!!')

    train_step_signature = [
    (
     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
     tf.TensorSpec(shape=(None, None), dtype=tf.int64)),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]
    
    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, labels):
        (inp, tar_inp) = inputs
        tar_real = labels
      
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp],
                                         training = True)
            loss = loss_function(tar_real, predictions)
      
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
      
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))


    with open(CATEGORY_FILE, 'r') as f:
        categories = f.read().splitlines() 

    class_preds = []
    for line in categories:
        line_arr = []
        trimmed_line = line.translate(str.maketrans('', '', string.whitespace)).split(',')
        for x in trimmed_line:
            if int(x) != tokenizer.cls_token_id and int(x) != tokenizer.sep_token_id:
                line_arr.append(int(x))
        class_preds.append(line_arr)
    del categories

    XA = []
    XB = []
    Y = []
    for pred_idx in range(HIST_LEN, len(class_preds)):
        if len(class_preds[pred_idx]) == 0:
            continue

        xa = []
        for i in reversed(range(1,HIST_LEN)):
            for cat in class_preds[pred_idx-i]:
                xa.append(cat)
        if len(xa) == 0:
            continue
        
        if len(xa) > INPUT_SIZE:
            xa = xa[len(xa)-INPUT_SIZE:]
        else:
            xa = xa + [tokenizer.pad_token_id] * INPUT_SIZE
            xa = xa[:INPUT_SIZE]

        xb = [tokenizer.cls_token_id] + class_preds[pred_idx]
        xb = xb + [tokenizer.pad_token_id] * (OUTPUT_SIZE)
        xb = xb[:OUTPUT_SIZE]

        y = class_preds[pred_idx].copy()
        y.append(tokenizer.sep_token_id)
        y = y + [tokenizer.pad_token_id] * (OUTPUT_SIZE)
        y = y[:OUTPUT_SIZE]

        XA.append(np.array(xa))
        XB.append(np.array(xb))
        Y.append(np.array(y))
    del class_preds

    for epoch in range(EPOCHS):
        XA_train, XA_test = train_test_split(XA, train_size=0.8, random_state=42)
        XB_train, XB_test = train_test_split(XB, train_size=0.8, random_state=42)
        y_train, y_test = train_test_split(Y, train_size=0.8, random_state=42)
        train_batches = []
        
        N_SPLITS = len(y_train) // 64
        for i in range(N_SPLITS-1):
            train_batches.append(((XA_train[i*64 : (i+1)*64], XB_train[i*64 : (i+1)*64]), y_train[i*64 : (i+1)*64]))
        
        print(f"Number of batches: {len(train_batches)}")
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (inp, tar)) in enumerate(train_batches):
            try:
                train_step(inp, tar)
            except Exception as exception:
                print(exception)
            if batch % 50 == 0:
              print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        if (epoch + 1) % 1 == 0:
              ckpt_save_path = ckpt_manager.save()
              print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


    os.mkdir(MODEL_DIR)
    transformer.save(MODEL_DIR)


def main(args):      
    train_split(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Epochs", default=100)
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.1)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-6)
    parser.add_argument("--history-length", type=int, help="Number of words as history", default=50)
    parser.add_argument("--pred-length", type=int, help="Number of words as history", default=10)
    parser.add_argument("--classes", type=int, help="Number of classes: length of bag of words + 1", default=1480)
    parser.add_argument("--splits", type=int, help="Number of data splits", default=75)
    parser.add_argument("--steps", type=int, help="Number of training steps", default=1000)
    parser.add_argument("--folder", type=str, help="Name for model folder", required=True)
    parser.add_argument("--file", type=str, help="Path to category data file", required=True)
    args = parser.parse_args()
    main(args)

            