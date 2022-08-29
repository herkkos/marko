import random

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization, LSTM, PReLU, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


REUSES = 5
BATCH_SIZE = 128
EPOCHS = 50
HISTORY_VAR = 5
PRED_LENGTH = 15
N_CATEGORIES = 3205
N_SPLITS = 50

RESPONSE_MODEL = 'marko_response_medium.h5'
C_FILE = 'categories_medium.txt'
CATEGORY_BATCH_SIZE = 5000

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'


def main():      
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

    with open(C_FILE, 'r') as f:
        categories = f.readlines()

    class_preds = []
    for line in categories:
        if line.strip():
            class_preds.append([int(x) for x in line.strip().split(',')])

    split_array = list(range(N_SPLITS))
    random.shuffle(split_array)
    split_size = len(class_preds) // N_SPLITS
    for split_idx in split_array:
        split_preds = class_preds[split_idx*split_size : ((split_idx+1)*split_size) - 1]

        XA = [] # Previous messages
        XB = [] # New words one-hot
        y = []
        for pred_idx in range(HISTORY_VAR, len(split_preds) - 1):
            category_mat = []
            for prev_category_idx in reversed(range(1, HISTORY_VAR + 1)):
                classes_one_hot = np.zeros(N_CATEGORIES)
                for _class in split_preds[pred_idx-prev_category_idx]:
                    classes_one_hot[_class] = 1
                category_mat.append(classes_one_hot)

            xx = np.array([[0] * (N_CATEGORIES - 1) + [1]] * PRED_LENGTH)
            for cat_idx, cat in enumerate(split_preds[pred_idx][:PRED_LENGTH]):
                XA.append(category_mat)
                XB.append(xx)
                y.append(to_categorical(cat, N_CATEGORIES))
                xx[-(cat_idx + 1)] = to_categorical(cat, N_CATEGORIES)

        XA = np.array(XA, dtype=np.float32)
        XB = np.array(XB).astype(np.float32)
        y = np.array(y).astype(np.float32)

        # TODO: validation set & outlier analysis -> outlier deletion because they probably contain silly msgs

        model.fit(x=[XA, XB],
                  y=y,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  shuffle=True)
    
    model.save(RESPONSE_MODEL)

if __name__ == '__main__':
    main()
    