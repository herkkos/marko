import argparse
import json
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate, PReLU, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam


EMPTY_CHAR = '#'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö0123456789:❤👍😂😆😩🤣"()-:;,!?.' + EMPTY_CHAR
N_CHARS = len(CHARS)
CHAR_TO_INT = dict((w, i) for i, w in enumerate(CHARS))


def create_model(args):
    inputA = Input(shape=(args.history_length))
    inputB = Input(shape=(args.pred_length))
    # Categories
    x = Embedding(input_dim=args.classes, output_dim=64)(inputA)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    # x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    # x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Model(inputs=inputA, outputs=x)
    # Chars
    y = Embedding(input_dim=N_CHARS, output_dim=64)(inputB)
    y = Bidirectional(LSTM(64, return_sequences=True))(y)
    y = BatchNormalization()(y)
    # y = PReLU()(y)
    y = Dropout(0.05)(y)
    y = Bidirectional(LSTM(64, return_sequences=False))(y)
    y = BatchNormalization()(y)
    # y = PReLU()(y)
    y = Dropout(0.05)(y)    
    y = Model(inputB, outputs=y)
    # Combine
    combined = Concatenate()([x.output, y.output])
    z = Dense(128)(combined)
    z = BatchNormalization()(z)
    z = PReLU()(z)
    z = Dropout(args.dropout)(z)
    z = Dense(N_CHARS, activation='softmax')(z)
    # Model
    model = Model(inputs=[x.input, y.input], outputs=z)
    opt = Adam(learning_rate=args.lr)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model

def find_newest_model(path):
    print(f"Finding newest model: {path}")
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def load_clf_model(args):
    print(f"Loading model: {args.folder}")
    if not os.path.isdir(args.folder):
        os.mkdir(args.folder)
        
    try:
        filename = find_newest_model(args.folder)
        model = load_model(filename)
        opt = Adam(learning_rate=args.lr)
        model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
        print(f"Loaded newest model: {args.folder}")
    except Exception as e:
        print(e)
        model = create_model(args)
        print("Training new model from scratch")

    return model

def train_split(args, split_idx: int, split_size: int):
    print(f"Training random index: {split_idx}")
    model = load_clf_model(args)
    
    with open(args.message_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    messages = [x['generation'] for x in data]
    
    split_messages = messages[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del messages, data
    
    with open(args.file, 'r') as f:
        categories = f.readlines()
        
    split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del categories
    
    class_preds = []
    for line in split_categories:
        if line.strip():
            class_preds.append([int(x) for x in line.strip().split(',')])
        else:
            class_preds.append([random.randint(0, args.classes-1)])
    del split_categories
    
    XA = [] # Categories for message
    XB = [] # Previous characters of the message
    y = []
    for msg_idx, msg in enumerate(split_messages):
        msg_str = msg.strip() + EMPTY_CHAR
        
        xa = []
        for word in class_preds[msg_idx][:args.history_length]:
            xa.append(word)
        xa = [args.classes - 1] * args.history_length + xa
        xa = xa[-args.history_length:]
        
        for char_idx, char in enumerate(msg_str[:-1]):
            xx = []
            for i in reversed(range(char_idx + 1)):
                xx.append(CHAR_TO_INT[msg_str[char_idx-i]])
            xx = [CHAR_TO_INT[EMPTY_CHAR]] * args.pred_length + xx
            xx = xx[-args.pred_length:]
        
            XA.append(xa)
            XB.append(xx)
            y.append(to_categorical(CHAR_TO_INT[msg_str[char_idx + 1]], N_CHARS))

    XA = np.array(XA).astype(float)
    XB = np.array(XB).astype(float)
    y = np.array(y).astype(float)
        
    XA_train, XA_test = train_test_split(XA, train_size=0.7, random_state=42)
    XB_train, XB_test = train_test_split(XB, train_size=0.7, random_state=42)
    y_train, y_test = train_test_split(y, train_size=0.7, random_state=42)
    
    model.fit(x=[XA_train, XB_train],
              y=y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=([XA_test,XB_test], y_test),
              callbacks=[EarlyStopping(monitor='val_loss', patience=args.early_stop),
                         ReduceLROnPlateau('val_loss', 0.5, patience=args.reduce_lr)],
              shuffle=True)
    
    model.save(os.path.join(args.folder, "word_gen.h5"))

def main(args):    
    with open(args.file, 'r') as f:
        categories = f.readlines()

    split_size = len(categories) // args.splits
    del categories

    for epoch in range(args.steps):
        print(f"Training random split: {epoch}")
        try:
            train_split(args, random.randint(0, args.splits-1), split_size)
        except:
            print("GPU memory full")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Epochs", default=50)
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.33)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-6)
    parser.add_argument("--early-stop", type=int, help="earlystop", default=10)
    parser.add_argument("--reduce-lr", type=int, help="reduce-lr", default=5)
    parser.add_argument("--history-length", type=int, help="Number of words as history", default=10)
    parser.add_argument("--pred-length", type=int, help="Number of words as history", default=50)
    parser.add_argument("--classes", type=int, help="Number of classes: length of bag of words", default=13405)
    parser.add_argument("--splits", type=int, help="Number of data splits", default=200)
    parser.add_argument("--steps", type=int, help="Number of training steps", default=50)
    parser.add_argument("--message-file", type=str, help="Path to message file", default="../X_data.json")
    parser.add_argument("--folder", type=str, help="Name for model folder", required=True)
    parser.add_argument("--file", type=str, help="Path to category data file", required=True)
    args = parser.parse_args()
    main(args)
    


