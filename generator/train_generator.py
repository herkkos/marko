import argparse
import json
import os
import random

from numba import cuda
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate, PReLU
from tensorflow.keras.optimizers import Adam


BATCH_SIZE = 128
EPOCHS = 50
HISTORY_VAR = 5
N_CATEGORIES = 3205
PRED_LENGTH = 160
N_SPLITS = 150
TRAIN_SPLITS = 50

X_FILE = '../X_data.json'
C_FILE = '../categories_medium.txt'
MODEL_NAME = 'marko_gene_medium_2.h5'

EMPTY_CHAR = '#'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö.' + EMPTY_CHAR
N_CHARS = len(CHARS)
CHAR_TO_INT = dict((w, i) for i, w in enumerate(CHARS))


def create_model():
    inputA = Input(shape=(N_CATEGORIES))
    inputB = Input(shape=(PRED_LENGTH, N_CHARS))
    # Categories
    x = Dense(32)(inputA)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.05)(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.05)(x)
    x = Model(inputs=inputA, outputs=x)
    # Chars
    y = LSTM(96, return_sequences=True)(inputB)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Dropout(0.2)(y)
    y = LSTM(96, return_sequences=False)(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Dropout(0.2)(y)    
    y = Model(inputB, outputs=y)
    # Combine
    combined = Concatenate()([x.output, y.output])
    z = Dense(128)(combined)
    z = BatchNormalization()(z)
    z = PReLU()(z)
    z = Dropout(0.05)(z)
    z = Dense(128)(z)
    z = BatchNormalization()(z)
    z = PReLU()(z)
    z = Dropout(0.05)(z)
    z = Dense(N_CHARS, activation='softmax')(z)
    # Model
    model = Model(inputs=[x.input, y.input], outputs=z)
    opt = Adam(learning_rate=0.00001)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model

def find_newest_model(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def load_clf_model(args):
    filename = find_newest_model(args.folder)
    try:
        model = load_model(filename)
        print(f"Loaded newest model: {args.model_name}")
    except:
        model = create_model(args)
        print("Training new model from scratch")

    return model

def train_split(split_idx: int, split_size: int):
    print(f"Training random index: {split_idx}")
    model = load_clf_model(MODEL_NAME)
    
    with open(X_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    messages = [x['formated'] for x in data]
    
    split_messages = messages[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del messages, data
    
    with open(C_FILE, 'r') as f:
        categories = f.readlines()
        
    split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del categories
    
    class_preds = []
    for line in split_categories:
        if line.strip():
            class_preds.append([int(x) for x in line.strip().split(',')])
        else:
            class_preds.append([])
    
    XA = [] # Categories for message
    XB = [] # Previous characters of the message
    y = []
    for msg_idx, msg in enumerate(split_messages):
        msg_str = EMPTY_CHAR*PRED_LENGTH + msg.strip() + EMPTY_CHAR
        
        classes_one_hot = np.zeros(N_CATEGORIES)
        for _class in class_preds[msg_idx]:
            classes_one_hot[_class] = 1
        
        for char_idx, char in enumerate(msg_str[:-1]):
            if char_idx < PRED_LENGTH - 1:
                continue

            xx = []
            for i in reversed(range(PRED_LENGTH)):
                xx.append(to_categorical(CHAR_TO_INT[msg_str[char_idx-i]], N_CHARS))
            
            if (not class_preds[msg_idx]):
                continue
            
            if max(class_preds[msg_idx]) == 0:
                continue
            XA.append(classes_one_hot)
            XB.append(xx)
            y.append(to_categorical(CHAR_TO_INT[msg_str[char_idx + 1]], N_CHARS))

    XA = np.array(XA).astype(float)
    XB = np.array(XB).astype(float)
    y = np.array(y).astype(float)
        
    model.fit(x=[XA, XB],
              y=y,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True)
    
    model.save(MODEL_NAME)

def main(args):    
    with open(C_FILE, 'r') as f:
        categories = f.readlines()

    split_size = len(categories) // N_SPLITS
    del categories

    for epoch in range(TRAIN_SPLITS):
        print(f"Training random split: {epoch}")
        try:
            train_split(random.randint(0, N_SPLITS-1), split_size)
        except:
            print("GPU memory full")
            device = cuda.get_current_device()
            device.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Epochs", default=50)
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.15)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--pred-length", type=int, help="Number of words as history", default=30)
    parser.add_argument("--classes", type=int, help="Number of classes: length of bag of words + 1", default=3205)
    parser.add_argument("--splits", type=int, help="Number of data splits", default=125)
    parser.add_argument("--steps", type=int, help="Number of training steps", default=50)
    parser.add_argument("--folder", type=str, help="Name for model folder", required=True)
    parser.add_argument("--file", type=str, help="Path to category data file", required=True)
    args = parser.parse_args()
    main(args)
    

    HISTORY_VAR = 5
    N_CATEGORIES = 3205
    PRED_LENGTH = 160
    N_SPLITS = 150
    TRAIN_SPLITS = 50

    X_FILE = '../X_data.json'
    C_FILE = '../categories_medium.txt'
    MODEL_NAME = 'marko_gene_medium_2.h5'
