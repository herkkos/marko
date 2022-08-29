import csv
import json
import os
import random

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from utils import parse_args

REUSES = 3
BATCH_SIZE = 128
EPOCHS = 50
N_CATEGORIES = 13405
PRED_LENGTH = 15
N_SPLITS = 5

X_FILE = 'X_data.json'
C_FOLDER = 'categories_large'
MODEL_NAME = 'marko_gen_large.h5'

EMPTY_CHAR = '#'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö.' + EMPTY_CHAR

CATEGORY_BATCH_SIZE = 5000

FILES = ['0.csv',
         '1.csv',
         '2.csv',
         '3.csv',
         '4.csv',
         '5.csv',
         '6.csv',
         '7.csv',
         '8.csv',
         '9.csv',
         '10.csv',
         '11.csv',
         '12.csv',
         '13.csv',
         '14.csv',
         '15.csv',
         '16.csv',
         '17.csv',
         '18.csv',
         '19.csv',
         '20.csv',
         '21.csv',
         '22.csv',
         '23.csv',
         '24.csv',
         '25.csv',
         '26.csv',
         '27.csv',
         '28.csv',
         '29.csv',
         '30.csv',
         '31.csv',
         '32.csv',
         '33.csv',
         '34.csv',
         '35.csv',
         '36.csv',
         '37.csv',
         '38.csv',
         '39.csv'
         ]

def main():    
    with open(X_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    messages = [x['formated'] for x in data]

    n_chars = len(CHARS)

    char_to_int = dict((w, i) for i, w in enumerate(CHARS))
    int_to_char = dict((i, w) for i, w in enumerate(CHARS))

### KAKSI INPUTTIA
    # Input
    inputA = Input(shape=(N_CATEGORIES,))
    inputB = Input(shape=(PRED_LENGTH, n_chars))
    # Categories
    x = Dense(100, activation='relu')(inputA)
    x = BatchNormalization()(x)
    x = Model(inputs=inputA, outputs=x)
    # Chars
    y = LSTM(PRED_LENGTH * n_chars + 1, activation='relu', return_sequences=True)(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)
    y = LSTM(PRED_LENGTH * n_chars + 1, activation='relu', return_sequences=True)(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y) 
    y = LSTM(PRED_LENGTH * n_chars + 1, activation='relu', return_sequences=False)(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)
    y = Model(inputB, outputs=y)
    # Combine
    combined = Concatenate()([x.output, y.output])    
    z = Dense(n_chars, activation='relu')(combined)
    z = Dense(n_chars, activation='softmax')(z)
    # Model
    model = Model(inputs=[x.input, y.input], outputs=z)
    opt = Adam(learning_rate=0.00005)
    # model.compile(opt, 'mse', metrics=['accuracy'])
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    
    file_idx = 0
    for reuse_idx in range(REUSES):
        files = os.listdir(C_FOLDER)
        random.shuffle(files)
        for filename in files:
            with open(os.path.join(C_FOLDER, str(filename))) as f:
                reader = csv.reader(f , delimiter=',')
                categories = list(reader)
            categories = np.array(categories).astype(float)
            
            XA = []
            XB = []
            y = []
            for msg_idx, msg in enumerate(messages[file_idx*CATEGORY_BATCH_SIZE : (file_idx + 1)*CATEGORY_BATCH_SIZE]):
                msg_str = EMPTY_CHAR*PRED_LENGTH + msg.strip()
                for char_idx, char in enumerate(msg_str[:-1]):
                    if char_idx < PRED_LENGTH - 1:
                        continue

                    xx = []
                    for i in reversed(range(PRED_LENGTH)):
                        xx.append(to_categorical(char_to_int[msg_str[char_idx-i]], n_chars))
                    
                    if len(categories[msg_idx]) != N_CATEGORIES:
                        continue
                    if max(categories[msg_idx]) == 0:
                        continue
                    XA.append(categories[msg_idx])
                    XB.append(xx)
                    y.append(to_categorical(char_to_int[msg_str[char_idx + 1]], n_chars))

            XA = np.array(XA).astype(float)
            XB = np.array(XB).astype(float)
            y = np.array(y).astype(float)

            split_size = len(XA) // N_SPLITS
            for split_idx in range(N_SPLITS):
                XA_split = XA[split_idx*split_size : ((split_idx+1)*split_size) - 1]
                XB_split = XB[split_idx*split_size : ((split_idx+1)*split_size) - 1]
                y_split = y[split_idx*split_size : ((split_idx+1)*split_size) - 1]
                
                model.fit(x=[XA_split, XB_split],
                          y=y_split,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

        if file_idx < len(FILES):
            file_idx = file_idx + 1
        else:
            file_idx = 0
        
    model.save(MODEL_NAME)


if __name__ == '__main__':
    args = parse_args()
    main()
