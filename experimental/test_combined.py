import argparse
import json
import os
import random

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


EMPTY_CHAR = '#'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö0123456789:❤👍😂😆😩🤣"()-:;,!?.' + EMPTY_CHAR
N_CHARS = len(CHARS)
CHAR_TO_INT = dict((w, i) for i, w in enumerate(CHARS))
INT_TO_CHAR = dict((i, w) for i, w in enumerate(CHARS))

FOLDER = 'combined1'
CAT_FILE = '../categories_1000.txt'
N_CLASSES = 1479
PRED_LENGTH = 160
HIST_LEN = 45


def find_newest_model(path):
    print(f"Finding newest model: {path}")
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def load_clf_model():
    print(f"Loading model: {''}")
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
        
    try:
        filename = find_newest_model(FOLDER)
        model = load_model(filename)
        print(f"Loaded newest model: {FOLDER}")
    except Exception as e:
        print(e)

    return model

def train_split(args, split_idx: int, split_size: int):
    print(f"Training random index: {split_idx}")
    model = load_clf_model()
    
    with open('../X_data.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    messages = [x['generation'] for x in data]
    
    split_messages = messages[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del messages, data
    
    with open(CAT_FILE, 'r') as f:
        categories = f.readlines()
        
    split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del categories
    
    class_preds = []
    for line in split_categories:
        line_arr = []
        if line.strip():
            for x in line.strip().split(','):
                line_arr.append(int(x))
            line_arr.append(N_CLASSES)
        class_preds.append(line_arr)
    del split_categories

    XA = [] # Old words one-hot
    XB = [] # New chars one-hot
    y = []
    prevs = []
    for idx in range(0, len(class_preds)):
        if (not class_preds[idx]):
            continue
        
        msg_str = EMPTY_CHAR*PRED_LENGTH + split_messages[idx].strip() + EMPTY_CHAR
        for cat in class_preds[idx]:
            if cat != (N_CLASSES):
                prevs.append(cat)

        if len(prevs) < HIST_LEN:
            continue

        prevs = prevs[-HIST_LEN:]
        xa = []
        for i in range(0, HIST_LEN):
            xa.append(to_categorical(prevs[i], N_CLASSES))

        for char_idx, char in enumerate(msg_str[:-1]):
            if char_idx < PRED_LENGTH - 1:
                continue

            xx = []
            for i in reversed(range(PRED_LENGTH)):
                xx.append(to_categorical(CHAR_TO_INT[msg_str[char_idx-i]], N_CHARS))
            
            XA.append(xa)
            XB.append(xx)
            y.append(to_categorical(CHAR_TO_INT[msg_str[char_idx + 1]], N_CHARS))

    XA = np.array(XA, dtype=np.float32)
    XB = np.array(XB, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return_str = ''
    for i in range(len(y)):
        XA_test = np.expand_dims(XA[i], 0)
        XB_test = np.expand_dims(XB[i], 0)
        score = model.predict([XA_test, XB_test])
        return_str += INT_TO_CHAR[score.argmax()]
        print(INT_TO_CHAR[y[i].argmax()] + " : " + INT_TO_CHAR[score.argmax()])

def main(args):      
    with open(CAT_FILE, 'r') as f:
        categories = f.readlines()
    
    split_size = len(categories) // 2000
    del categories
   
    for epoch in range(1):
        print(f"Training random split: {epoch}")
        try:
            train_split(args, random.randint(0, 2000-1), split_size)
        except Exception as e: 
            print(e)
            print("GPU memory full")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--history-length", type=int, help="Number of words as history", default=45)
    parser.add_argument("--pred-length", type=int, help="Number of words as history", default=160)
    parser.add_argument("--classes", type=int, help="Number of classes: length of bag of words", default=1479)
    parser.add_argument("--splits", type=int, help="Number of data splits", default=500)
    parser.add_argument("--steps", type=int, help="Number of training steps", default=50)
    parser.add_argument("--file", type=str, help="Path to category data file", required=True)
    parser.add_argument("--message-file", type=str, help="Path to message file", default="../X_data.json")
    parser.add_argument("--folder", type=str, help="Name for model folder", required=True)
    args = parser.parse_args()
    main(args)

            