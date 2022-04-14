import csv
import json
import numpy as np
import pickle
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from utils import parse_args

N_SPLITS =  9
REUSES = 5
BATCH_SIZE = 128
EPOCHS = 50

MODEL_NAME = 'marko_pred_1000.h5'
C_FILE = 'categories1000.csv'
X_FILE = 'X_data.json'
PRED_FILE = 'class_preds1000'
CHAR_FILE = 'chars1000'
MAX_LENGTH = 160
WORD_LENGTH = 16


CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'


def main():
    n_chars = len(CHARS)
    char_to_int = dict((w, i) for i, w in enumerate(CHARS))
    
    with open(C_FILE, 'r') as f:
        reader = csv.reader(f , delimiter=',')
        categories = list(reader)
    categories = np.array(categories).astype(float)
    
    with open(X_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    n_labels = categories.shape[1]
    p = min((n_labels*3) / len(data), 1)
    
    y = []
    X = []
    for idx, message in enumerate(data):
        msg_text = message['formated']
        if msg_text == '':
            continue
        if max(categories[idx]) > 0 or random.random() < p:
            if len(msg_text) > MAX_LENGTH:
                continue
            for word in msg_text.split():
                if len(word) > WORD_LENGTH:
                    continue
                y.append(categories[idx])
                X.append(word.rjust(WORD_LENGTH))

    
    n_patterns = len(X)
    print("Total Patterns: "+ str(n_patterns))

    new_X = np.zeros((len(X), WORD_LENGTH, n_chars))
    for x_idx, x in enumerate(X):
        new_x = []
        for idx, char in enumerate(x):
            new_x.append(char_to_int[char])
        new_X[x_idx] = to_categorical(new_x, n_chars)

    y = np.array(y).astype(float)
    
    model = Sequential()
    model.add(LSTM(160, input_shape=(WORD_LENGTH, n_chars), return_sequences=True))
    model.add(BatchNormalization(scale=False))
    model.add(Dropout(0.3))
    model.add(LSTM(160, return_sequences=True))
    model.add(BatchNormalization(scale=False))
    model.add(Dropout(0.3))
    model.add(LSTM(160))
    model.add(BatchNormalization(scale=False))
    model.add(Dropout(0.3))
    model.add(Dense(160, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(categories.shape[1], activation='relu'))
    opt = Adam(learning_rate=0.0001)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
    
    split_size = len(new_X) // N_SPLITS
    for ite in range(REUSES):
        for i in range(N_SPLITS):
            print(f'Training iteration {ite+1} with split {i+1}')
            X_split = new_X[i*split_size:((i+1)*split_size) - 1]
            y_split = y[i*split_size:((i+1)*split_size) - 1]
            model.fit(X_split,
                      y_split,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      shuffle=True
            )

    model.save(MODEL_NAME)

    class_preds = []
    for message in data:
        msg_words = message['formated'].split()
        if not msg_words:
            class_preds.append(np.zeros((1, n_labels)))
            continue
        X = np.zeros((len(msg_words), WORD_LENGTH, n_chars))
    
        for idx, word in enumerate(msg_words):
            x = []
            for char in word[0:WORD_LENGTH].rjust(WORD_LENGTH):
                x.append(char_to_int[char])
            X[idx] = to_categorical(x, n_chars)

        pred = model.predict(X)
        pred = sum(pred, 0)
        class_preds.append(pred.tolist())
        
    with open(PRED_FILE, 'wb') as fp:
        pickle.dump(class_preds, fp)
        
        
if __name__ == '__main__':
    args = parse_args()
    main()
    