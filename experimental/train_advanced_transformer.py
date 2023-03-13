'''
This training script is based on training TensorFlow tutorial which can be found in:
https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb

Copyright 2022 The TensorFlow Authors.
https://www.apache.org/licenses/LICENSE-2.0
'''

import argparse
from datetime import datetime
from random import randint, random
import string

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from transformers import BertTokenizer

import advanced_transformer

tokenizer = BertTokenizer.from_pretrained('bert-2-vocab.txt')
N_CLASSES = tokenizer.vocab_size

MODEL_DIR = 'advanced5.h5'
CATEGORY_FILE = '../categories_bert2_ts.txt'

HIST_LEN = 20
MAX_TIME = 60 * 30
OVERALL = 200

EPOCHS = 1000
B_SIZE = 16
learning_rate = 0.00005

SPEAKER_SIZE = 256
SECOND_SIZE = 256
OTHER_SIZE = 256
OVERALL_SIZE = 256
OUTPUT_SIZE = 128


def save_model(model, path):
    model.save_weights(path)
    print("model saved")

def _load_model(path):
    transformer = advanced_transformer.create_model()
    print("loading weights")
    transformer.load_weights(path)
    print("weights loaded")
    K.set_value(transformer.optimizer.learning_rate, learning_rate)
    print("LR set")
    return transformer


def train_model(args):
    print("loading model...")
    transformer = _load_model(MODEL_DIR)
    #transformer = advanced_transformer.create_model()

    print("loading categories...")
    with open(CATEGORY_FILE, 'r') as f:
        categories = f.read().splitlines()

    print("processing categories...")
    class_preds = []
    senders = []
    timestamps = []
    cat_counts = np.ones(N_CLASSES)
    for line in categories:
        line_arr = []
        trimmed_line = line.translate(str.maketrans('', '', string.whitespace)).split(',')
        sender = trimmed_line[0]
        timestamp = int(datetime.fromisoformat(trimmed_line[1]).timestamp())
        
        for x in trimmed_line[2:]:
            if int(x) != tokenizer.cls_token_id and int(x) != tokenizer.sep_token_id:
                cat_counts[int(x)] += 1
                line_arr.append(int(x))
        class_preds.append(line_arr)
        senders.append(sender)
        timestamps.append(timestamp)
    cat_counts = cat_counts / len(categories)
    del categories

    print("processing training data...")
    X_speaker = []
    X_second = []
    X_other = []
    X_time = []
    X_overall = []
    XB = []
    Y = []
    prev_ts = timestamps[0]
    msgs_since = 0
    for pred_idx in range(1, len(class_preds)):
        # Discard empty target messages
        if len(class_preds[pred_idx]) == 0:
            continue
        
        # Normalize training data based on how common the output is
        y = class_preds[pred_idx].copy()
        if random() < sum(cat_counts[x] for x in y):
            continue
        y.append(tokenizer.sep_token_id)
        y = y + [tokenizer.pad_token_id] * (OUTPUT_SIZE)
        y = y[:OUTPUT_SIZE]
        
        # Time difference
        if prev_ts:
            # Chech if timestamp is invalid. Happens when conversation changes
            if timestamps[pred_idx] < prev_ts:
                prev_ts = timestamps[pred_idx]
                msgs_since = 0
                continue
            time_diff = min(timestamps[pred_idx] - timestamps[pred_idx - 1], MAX_TIME)
            msgs_since += 1
        
        # Collect sender, second and other categories
        sender = senders[pred_idx]
        second = senders[pred_idx - 1]
        sender_x = []
        second_x = []
        other_x = []
        for i in range(1, HIST_LEN + 1):
            h_i = pred_idx - i
            if msgs_since - i < 0:
                for j in range(i, HIST_LEN + 1):
                    sender_x.append([tokenizer.pad_token_id] * SPEAKER_SIZE)
                    second_x.append([tokenizer.pad_token_id] * SECOND_SIZE)
                    other_x.append([tokenizer.pad_token_id] * OTHER_SIZE)
                break
            if senders[h_i] == sender:
                sender_x.append((class_preds[h_i] + [tokenizer.pad_token_id] * SPEAKER_SIZE)[:SPEAKER_SIZE])
                second_x.append([tokenizer.pad_token_id] * SECOND_SIZE)
                other_x.append([tokenizer.pad_token_id] * OTHER_SIZE)
            elif senders[h_i] == second:
                sender_x.append([tokenizer.pad_token_id] * SPEAKER_SIZE)
                second_x.append((class_preds[h_i] + [tokenizer.pad_token_id] * SECOND_SIZE)[:SECOND_SIZE])
                other_x.append([tokenizer.pad_token_id] * OTHER_SIZE)
            else:
                sender_x.append([tokenizer.pad_token_id] * SPEAKER_SIZE)
                second_x.append([tokenizer.pad_token_id] * SECOND_SIZE)
                other_x.append((class_preds[h_i] + [tokenizer.pad_token_id] * OTHER_SIZE)[:OTHER_SIZE])

        # Overall
        overall_x = np.zeros(N_CLASSES)
        for i in range(1, OVERALL):
            h_i = pred_idx - i
            if msgs_since - i < 0:
                break
            for c in class_preds[h_i]:
                overall_x[c] += max(0.5, (OVERALL - i) / OVERALL)
        overall_x = overall_x / cat_counts
        # overall_x = overall_x / overall_x.sum()
        # ind = np.argpartition(overall_x, -OVERALL_SIZE)[-OVERALL_SIZE:].tolist()
        # overall_x = ind + [tokenizer.pad_token_id] * (OVERALL_SIZE)
        # overall_x = overall_x[:OVERALL_SIZE]

        xb = [tokenizer.cls_token_id] + class_preds[pred_idx]
        xb = xb + [tokenizer.pad_token_id] * (OUTPUT_SIZE)
        xb = xb[:OUTPUT_SIZE]

        X_speaker.append(np.array(sender_x, dtype=np.float32))
        X_second.append(np.array(second_x, dtype=np.float32))
        X_other.append(np.array(other_x, dtype=np.float32))
        X_time.append(np.array([time_diff], dtype=np.float32))
        X_overall.append(np.array(overall_x, dtype=np.float32))
        XB.append(np.array(xb, dtype=np.int64))
        Y.append(np.array(y, dtype=np.int64))

    print("splitting data...")
    random_state = randint(0, 100000)
    train_size=0.9
    X_speaker_train, X_speaker_test = train_test_split(np.array(X_speaker), train_size=train_size, random_state=random_state)
    X_second_train, X_second_test = train_test_split(np.array(X_second), train_size=train_size, random_state=random_state)
    X_other_train, X_other_test = train_test_split(np.array(X_other), train_size=train_size, random_state=random_state)
    X_time_train, X_time_test = train_test_split(np.array(X_time), train_size=train_size, random_state=random_state)
    X_overall_train, X_overall_test = train_test_split(np.array(X_overall), train_size=train_size, random_state=random_state)
    XB_train, XB_test = train_test_split(np.array(XB), train_size=train_size, random_state=random_state)
    y_train, y_test = train_test_split(np.array(Y), train_size=train_size, random_state=random_state)

    print("split data...")
    train_batches = []
    N_SPLITS = len(y_train) // B_SIZE
    for i in range(N_SPLITS-1):
        train_batches.append(((X_speaker_train[i*B_SIZE : (i+1)*B_SIZE],
                                X_second_train[i*B_SIZE : (i+1)*B_SIZE],
                                X_other_train[i*B_SIZE : (i+1)*B_SIZE],
                                X_time_train[i*B_SIZE : (i+1)*B_SIZE],
                                X_overall_train[i*B_SIZE : (i+1)*B_SIZE],
                                XB_train[i*B_SIZE : (i+1)*B_SIZE]),
                                y_train[i*B_SIZE : (i+1)*B_SIZE]))

    print("training...")
    for epch in range(EPOCHS):
        number_of_exceptions = 0
        for (batch, (train, tar)) in enumerate(train_batches):
            print(f"Epoch: {epch} batch: {batch}/{N_SPLITS}")
            try:
                transformer.fit(x=train,
                          y=tar,
                          epochs=25,
                          batch_size=B_SIZE,
                          shuffle=True)
                if batch % 50 == 0:
                    save_model(transformer, MODEL_DIR)
            except Exception as exception:
                tete = exception
                print(tete)
                number_of_exceptions += 1
                if number_of_exceptions % 3 == 0:
                    save_model(transformer, MODEL_DIR)
                    break

    save_model(transformer, MODEL_DIR)


def main(args):      
    train_model(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--model", type=str, help="Name for model folder", default='advanced')
    parser.add_argument("--checkpoint", type=str, help="Name for model folder", default='./checkpoints_advanced/train')
    parser.add_argument("--categories", type=str, help="Name for model folder", default='../categories_bert.txt')
    parser.add_argument("--history-length", type=int, help="Number of words as history", default=5)
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.1)
    parser.add_argument("--epochs", type=int, help="Epochs", default=1000)
    parser.add_argument("--lr", type=float, help="Dropout", default=0.00001)
    args = parser.parse_args()
    main(args)


