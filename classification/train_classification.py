import argparse
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization, LSTM, PReLU, Input, Concatenate, Bidirectional, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def create_model(args):
    print("Creating new model")
    # Input
    inputA = Input(shape=(args.history_length))
    inputB = Input(shape=(args.pred_length))
    # Words
    x = Embedding(input_dim=args.classes - 1, output_dim=256)(inputA)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Bidirectional(LSTM(256, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Model(inputs=inputA, outputs=x)
    
    y = Embedding(input_dim=args.classes, output_dim=256)(inputB)
    y = Bidirectional(LSTM(256, return_sequences=True))(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Dropout(args.dropout)(y)
    y = Bidirectional(LSTM(256, return_sequences=False))(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)
    y = Dropout(args.dropout)(y)
    y = Model(inputs=inputB, outputs=y)
    
    combined = Concatenate()([x.output, y.output])
    z = Dense(args.classes)(combined)
    z = BatchNormalization()(z)
    z = PReLU()(z)
    z = Dropout(args.dropout)(z)

    z = Dense(args.classes, activation='softmax')(z)
    # Model
    model = Model(inputs=[x.input, y.input], outputs=z)
    opt = Adam(learning_rate=args.lr)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
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
    
    with open(args.file, 'r') as f:
        categories = f.readlines()
        
    split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del categories
    
    class_preds = []
    for line in split_categories:
        line_arr = []
        if line.strip():
            for x in line.strip().split(','):
                line_arr.append(int(x))
            line_arr.append(args.classes - 1)
        class_preds.append(line_arr)
    del split_categories

    with open(args.corr, 'r') as f:
        corr_factors = f.readlines()
        
    for corr_idx, corr in enumerate(corr_factors):
        corr_factors[corr_idx] = max(0.0, float(corr))

    XA = [] # Old words one-hot
    XB = [] # New words one-hot
    y = []
    for pred_idx in range(0, len(class_preds)):
        if pred_idx < args.history_length:
            continue

        xa = []
        for i in reversed(range(1, args.history_length)):
            for cat in class_preds[pred_idx-i]:
                xa.append(cat)
        xa = [args.classes - 1] * args.history_length + xa
        xa = xa[-args.history_length:]
    
        for word_idx, word in enumerate(class_preds[pred_idx][:args.pred_length]):
            yy = to_categorical(word, args.classes)
            
            xb = []
            for i in reversed(range(word_idx)):
                xb.append(class_preds[pred_idx][word_idx-i])
            xb = [args.classes - 1] * args.pred_length + xb
            xb = xb[-args.pred_length:]
            
            XA.append(xa)
            XB.append(xb)
            y.append(yy)

    XA = np.array(XA, dtype=np.float32)
    XB = np.array(XB, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Number of samples: {len(y)}")

    XA_train, XA_test = train_test_split(XA, train_size=0.7, random_state=42)
    XB_train, XB_test = train_test_split(XB, train_size=0.7, random_state=42)
    y_train, y_test = train_test_split(y, train_size=0.7, random_state=42)
    
    model.fit(x=[XA_train, XB_train],
              y=y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=([XA_test,XB_test], y_test),
              callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                         ReduceLROnPlateau('loss', 0.5, patience=5)],
              shuffle=True)
    
    model.save(os.path.join(args.folder, "word_clf.h5"))

def main(args):      
    with open(args.file, 'r') as f:
        categories = f.readlines()
    
    split_size = len(categories) // args.splits
    del categories
   
    for epoch in range(args.steps):
        print(f"Training random split: {epoch}")
        try:
            train_split(args, random.randint(0, args.splits-1), split_size)
            # train_split(args, 0, split_size)
        except Exception as e: 
            print(e)
            print("GPU memory full")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Epochs", default=100)
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.33)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-6)
    parser.add_argument("--history-length", type=int, help="Number of words as history", default=20)
    parser.add_argument("--pred-length", type=int, help="Number of words as history", default=10)
    parser.add_argument("--classes", type=int, help="Number of classes: length of bag of words + 1", default=1480)
    parser.add_argument("--splits", type=int, help="Number of data splits", default=75)
    parser.add_argument("--steps", type=int, help="Number of training steps", default=1000)
    parser.add_argument("--folder", type=str, help="Name for model folder", required=True)
    parser.add_argument("--file", type=str, help="Path to category data file", required=True)
    parser.add_argument("--corr", type=str, help="Path to class correction file", required=True)
    args = parser.parse_args()
    main(args)

            