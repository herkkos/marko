import random

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization, LSTM, PReLU, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


BATCH_SIZE = 128
EPOCHS = 50
HISTORY_VAR = 5
PRED_LENGTH = 15
N_CATEGORIES = 3205
N_SPLITS = 100
TRAIN_SPLITS = 50

MODEL_NAME = 'marko_response_medium.h5'
C_FILE = '../categories_medium.txt'


def create_model():
    # Input
    inputA = Input(shape=(HISTORY_VAR, N_CATEGORIES,))
    inputB = Input(shape=(PRED_LENGTH, N_CATEGORIES))
    # Categories
    x = LSTM(16, return_sequences=False)(inputA)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.05)(x)
    x = Model(inputs=inputA, outputs=x)
    # Chars
    y = LSTM(16, activation='relu', return_sequences=True)(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.05)(y)
    y = LSTM(16, activation='relu', return_sequences=False)(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.05)(y)
    y = Model(inputB, outputs=y)
    # Combine
    combined = Concatenate()([x.output, y.output])    
    z = Dense(32, activation='relu')(combined)
    z = Dense(N_CATEGORIES, activation='softmax')(z)
    # Model
    model = Model(inputs=[x.input, y.input], outputs=z)
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
            class_preds.append([int(x) for x in line.strip().split(',')])
    
    XA = [] # Previous messages
    XB = [] # New words one-hot
    y = []
    for pred_idx in range(HISTORY_VAR, len(class_preds) - 1):
        category_mat = []
        for prev_category_idx in reversed(range(1, HISTORY_VAR + 1)):
            classes_one_hot = np.zeros(N_CATEGORIES)
            for _class in class_preds[pred_idx-prev_category_idx]:
                classes_one_hot[_class] = 1
            category_mat.append(classes_one_hot)

        xx = np.array([[0] * (N_CATEGORIES - 1) + [1]] * PRED_LENGTH)
        class_categories = class_preds[pred_idx][:PRED_LENGTH] + [N_CATEGORIES - 1]
        for cat_idx, cat in enumerate(class_categories):
            XA.append(category_mat)
            XB.append(xx)
            y.append(to_categorical(cat, N_CATEGORIES))
            xx = np.vstack([xx, to_categorical(cat, N_CATEGORIES)])[1:]

    XA = np.array(XA, dtype=np.float32)
    XB = np.array(XB).astype(np.float32)
    y = np.array(y).astype(np.float32)

    #TODO: validation set
    model.fit(x=[XA, XB],
              y=y,
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
    