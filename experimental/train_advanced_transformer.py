'''
This training script is based on training TensorFlow tutorial which can be found in:
https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb

Copyright 2022 The TensorFlow Authors.
https://www.apache.org/licenses/LICENSE-2.0
'''

import argparse
from datetime import datetime
from random import randint, random
from statistics import mean 
import string

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from transformers import BertTokenizer

import advanced_transformer

tokenizer = BertTokenizer.from_pretrained('bert-2-vocab.txt')
N_CLASSES = tokenizer.vocab_size

MODEL_DIR = 'advanced10.h5'
CATEGORY_FILE = '../categories_bert2_ts.txt'

HIST_LEN = 10
MAX_TIME = 60 * 30
OVERALL = 200

EPOCHS = 1000
B_SIZE = 32
learning_rate = 0.00005

INPUT_SIZE = 128
OVERALL_SIZE = 9999
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
    # transformer = _load_model(MODEL_DIR)
    transformer = advanced_transformer.create_model()

    print("loading categories...")
    with open(CATEGORY_FILE, 'r') as f:
        categories = f.read().splitlines()

    print("processing categories...")
    class_preds = []
    senders = []
    timestamps = []
    cat_counts = np.ones(OVERALL_SIZE)
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

    # sophisticated method for normalizing token counts
    cat_thres = cat_counts - cat_counts.min()
    lin_fac = 1.0 / cat_thres.max()
    cat_thres = cat_thres * lin_fac

    print("processing training data...")
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    X6 = []
    X7 = []
    X8 = []
    X9 = []
    X10 = []

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
        if random() < mean(cat_thres[x] for x in y):
            # print("Skip: ", tokenizer.decode(y, skip_special_tokens=True))
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
            msgs_since += 1

        if msgs_since < HIST_LEN:
            continue

        x2 = [(class_preds[pred_idx] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x1 = [(class_preds[pred_idx-1] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x3 = [(class_preds[pred_idx-2] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x4 = [(class_preds[pred_idx-3] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x5 = [(class_preds[pred_idx-4] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x6 = [(class_preds[pred_idx-5] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x7 = [(class_preds[pred_idx-6] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x9 = [(class_preds[pred_idx-7] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x8 = [(class_preds[pred_idx-8] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE] ]
        x10 = [(class_preds[pred_idx-9] + [tokenizer.pad_token_id] * INPUT_SIZE)[:INPUT_SIZE]]
        

        xb = [tokenizer.cls_token_id] + class_preds[pred_idx]
        xb = xb + [tokenizer.pad_token_id] * (OUTPUT_SIZE)
        xb = xb[:OUTPUT_SIZE]

        X1.append(np.array(x1, dtype=np.float32))
        X2.append(np.array(x2, dtype=np.float32))
        X3.append(np.array(x3, dtype=np.float32))
        X4.append(np.array(x4, dtype=np.float32))
        X5.append(np.array(x5, dtype=np.float32))
        X6.append(np.array(x6, dtype=np.float32))
        X7.append(np.array(x7, dtype=np.float32))
        X8.append(np.array(x8, dtype=np.float32))
        X9.append(np.array(x9, dtype=np.float32))
        X10.append(np.array(x10, dtype=np.float32))

        XB.append(np.array(xb, dtype=np.int64))
        Y.append(np.array(y, dtype=np.int64))

    print("splitting data...")
    random_state = randint(0, 100000)
    train_size=0.9
    X1_train, X1_test = train_test_split(np.array(X1), train_size=train_size, random_state=random_state)
    X2_train, X2_test = train_test_split(np.array(X2), train_size=train_size, random_state=random_state)
    X3_train, X3_test = train_test_split(np.array(X3), train_size=train_size, random_state=random_state)
    X4_train, X4_test = train_test_split(np.array(X4), train_size=train_size, random_state=random_state)
    X5_train, X5_test = train_test_split(np.array(X5), train_size=train_size, random_state=random_state)
    X6_train, X6_test = train_test_split(np.array(X6), train_size=train_size, random_state=random_state)
    X7_train, X7_test = train_test_split(np.array(X7), train_size=train_size, random_state=random_state)
    X8_train, X8_test = train_test_split(np.array(X8), train_size=train_size, random_state=random_state)
    X9_train, X9_test = train_test_split(np.array(X9), train_size=train_size, random_state=random_state)
    X10_train, X10_test = train_test_split(np.array(X10), train_size=train_size, random_state=random_state)
    XB_train, XB_test = train_test_split(np.array(XB), train_size=train_size, random_state=random_state)
    y_train, y_test = train_test_split(np.array(Y), train_size=train_size, random_state=random_state)

    print("split data...")
    train_batches = []
    N_SPLITS = len(y_train) // B_SIZE
    for i in range(N_SPLITS-1):
        train_batches.append(((
            X1_train[i*B_SIZE : (i+1)*B_SIZE],
            X2_train[i*B_SIZE : (i+1)*B_SIZE],
            X3_train[i*B_SIZE : (i+1)*B_SIZE],
            X4_train[i*B_SIZE : (i+1)*B_SIZE],
            X5_train[i*B_SIZE : (i+1)*B_SIZE],
            X6_train[i*B_SIZE : (i+1)*B_SIZE],
            X7_train[i*B_SIZE : (i+1)*B_SIZE],
            X8_train[i*B_SIZE : (i+1)*B_SIZE],
            X9_train[i*B_SIZE : (i+1)*B_SIZE],
            X10_train[i*B_SIZE : (i+1)*B_SIZE],
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
                          epochs=10,
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


