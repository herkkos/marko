import csv
import os
import random
from utils import parse_args

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization, LSTM, PReLU
from tensorflow.keras.optimizers import Adam

REUSES = 5
BATCH_SIZE = 64
EPOCHS = 120
HISTORY_VAR = 5
N_CATEGORIES = 3204

RESPONSE_MODEL = 'marko_response_medium.h5'
C_FOLDER = 'categories_medium'
CATEGORY_BATCH_SIZE = 5000

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'

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
    model = Sequential()
    model.add(LSTM(128, input_shape=(HISTORY_VAR, N_CATEGORIES), return_sequences=False))
    model.add(BatchNormalization(scale=False))
    model.add(PReLU())
    model.add(Dropout(0.05))
    
    model.add(Dense(N_CATEGORIES, activation='softmax'))
    opt = Adam(learning_rate=0.0001)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

    for reuse in range(REUSES):
        # TODO: stream files somehow
        random.shuffle(FILES)
        for filename in FILES:
            print(f"File: {filename}, reuse: {reuse}")
            with open(os.path.join(C_FOLDER, str(filename))) as f:
                reader = csv.reader(f , delimiter=',')
                categories = list(reader)
            class_preds = np.array(categories).astype(float)
            
            if (np.max(class_preds) < 1.0):  # for errors in classification
                continue
            
            X = []
            y = []
            for idx in range(HISTORY_VAR, len(class_preds) - 1):
                if max(class_preds[idx]) < 1.0:
                    continue
                x = []
                for i in reversed(range(1, HISTORY_VAR + 1)):
                    x.append(class_preds[idx-i])
                X.append(x)
                y.append(class_preds[idx].tolist())
        
            X = np.array(X).astype(float)
            y = np.array(y).astype(float)
            
            if (len(X) == 0 or len(y) == 0):
                continue
    
            model.fit(X,
                      y,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      shuffle=True)
    
    model.save(RESPONSE_MODEL)

if __name__ == '__main__':
    args = parse_args()
    main()
    
categories = []
for filename in FILES[5:10]:
    with open(os.path.join(C_FOLDER, str(filename))) as f:
        reader = csv.reader(f , delimiter=',')
        categories += list(reader)
