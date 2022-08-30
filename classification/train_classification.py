import argparse
import os
import random

from numba import cuda
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization, LSTM, PReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


MODEL_NAME = 'marko_response_simple.h5'
C_FILE = '../categories_medium.txt'


def create_model(args):
    print("Creating new model")
    # Input
    inputA = Input(shape=(args.pred_length, args.classes))
    # Words
    x = LSTM(30, return_sequences=True)(inputA)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = LSTM(30, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(args.dropout)(x)
    x = Dense(args.classes, activation='softmax')(x)
    # Model
    model = Model(inputs=inputA, outputs=x)
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
        print(f"Loaded newest model: {args.folder}")
    except Exception as e:
        print(e)
        model = create_model(args)
        print("Training new model from scratch")

    return model

def train_split(args, split_idx: int, split_size: int):
    print(f"Training random index: {split_idx}")
    model = load_clf_model(args)
    
    with open(C_FILE, 'r') as f:
        categories = f.readlines()
    
        
    split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del categories
    
    class_preds = []
    for line in split_categories:
        if line.strip():
            for x in line.strip().split(','):
                class_preds.append(x)
            class_preds.append(args.classes - 1)
    del split_categories
    
    X = [] # New words one-hot
    y = []
    xx = np.array([[0] * (args.classes - 1) + [1]] * args.pred_length)
    for pred_idx in range(0, len(class_preds) - 1):
        X.append(xx)
        yy = to_categorical(class_preds[pred_idx], args.classes)
        y.append(yy)
        xx = np.vstack([xx, yy])[1:]

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    #TODO: validation set
    model.fit(X,
              y,
              epochs=args.epochs,
              batch_size=args.batch_size,
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
        except:
            print("GPU memory full")
            device = cuda.get_current_device()
            device.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification model.")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
    parser.add_argument("--epochs", type=int, help="Epochs", default=50)
    parser.add_argument("--dropout", type=float, help="Dropout", default=0.15)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--pred-length", type=int, help="Number of words as history", default=30)
    parser.add_argument("--classes", type=int, help="Number of classes: length of bag of words + 1", default=3205)
    parser.add_argument("--splits", type=int, help="Number of data splits", default=1000)
    parser.add_argument("--steps", type=int, help="Number of training steps", default=200)
    parser.add_argument("--folder", type=str, help="Name for model folder", required=True)
    parser.add_argument("--file", type=str, help="Path to category data file", required=True)
    args = parser.parse_args()
    main(args)

            