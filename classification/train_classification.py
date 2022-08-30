import random

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization, LSTM, PReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


BATCH_SIZE = 32
EPOCHS = 200
PRED_LENGTH = 30
N_CATEGORIES = 3205
N_SPLITS = 125
TRAIN_SPLITS = 50

MODEL_NAME = 'marko_response_simple.h5'
C_FILE = '../categories_medium.txt'


def create_model():
    # Input
    inputA = Input(shape=(PRED_LENGTH, N_CATEGORIES))
    # Words
    x = LSTM(30, return_sequences=True)(inputA)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.15)(x)
    x = LSTM(30, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.15)(x)
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.15)(x)
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.15)(x)
    x = Dense(N_CATEGORIES, activation='softmax')(x)
    # Model
    model = Model(inputs=inputA, outputs=x)
    opt = Adam(learning_rate=0.0001)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    return model

def load_clf_model(model_name: str):
    #TODO: use folder instead of file
    #TODO: find newest model file from folder
    try:
        model = load_model(model_name)
        print(f"Loaded newest model: {model_name}")
    except:
        model = create_model()
        print("Training new model from scratch")

    return model

def train_split(split_idx: int, split_size: int):
    print(f"Training random index: {split_idx}")
    model = load_clf_model(MODEL_NAME)
    
    with open(C_FILE, 'r') as f:
        categories = f.readlines()
        
    split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
    del categories
    
    class_preds = []
    for line in split_categories:
        if line.strip():
            for x in line.strip().split(','):
                class_preds.append(x)
            class_preds.append(N_CATEGORIES - 1)
    del split_categories
    
    X = [] # New words one-hot
    y = []
    xx = np.array([[0] * (N_CATEGORIES - 1) + [1]] * PRED_LENGTH)
    for pred_idx in range(0, len(class_preds) - 1):
        X.append(xx)
        yy = to_categorical(class_preds[pred_idx], N_CATEGORIES)
        y.append(yy)
        xx = np.vstack([xx, yy])[1:]

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    #TODO: validation set
    model.fit(X,
              y,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True)
    
    model.save(MODEL_NAME)

def main():      
    with open(C_FILE, 'r') as f:
        categories = f.readlines()

    split_size = len(categories) // N_SPLITS
    del categories
    
    for epoch in range(TRAIN_SPLITS):
        print(f"Training random split: {epoch}")
        try:
            train_split(random.randint(0, N_SPLITS-1), split_size)
        except:
            ## TODO: clear GPU memory
            print("GPU memory full")

if __name__ == '__main__':
    #TODO: move constants to CLI params
    main()
            