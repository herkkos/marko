import csv
import json
import numpy as np
import pickle
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical, custom_object_scope
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from utils import parse_args

N_SPLITS =  9
REUSES = 5
BATCH_SIZE = 128
EPOCHS = 150

CLASS_MODEL = 'marko_pred_1000.h5'
RESPONSE_MODEL = 'marko_response_1000.h5'
C_FILE = 'categories1000.csv'
X_FILE = 'X_data.json'
PRED_FILE = 'class_preds1000'
MESSAGE_PRED = 'whole_messages1000'
CHAR_FILE = 'chars1000'
MAX_LENGTH = 160
WORD_LENGTH = 16
N_CATEGORIES = 1046

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'


def main():
    n_chars = len(CHARS)
    char_to_int = dict((w, i) for i, w in enumerate(CHARS))
    
    with open(X_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    custom_objects = {'<lambda>': relu_advanced}
    with custom_object_scope(custom_objects):
        classification_model = load_model(CLASS_MODEL)

    X = []
    msg_array = []
    last_sender = ''
    for msg in data:
        if msg['formated'].strip() == '':
            continue
        if msg['sender'] == last_sender:
            for word in msg['formated'].split():
                msg_array.append(word)
        else:
            X.append(msg_array)
            msg_array = msg['formated'].split()
            last_sender = msg['sender']
    messages=X[1:]
    
    with open('X.txt', 'w') as f:
        for msg in messages:
            msg_text = ' '.join(msg)+'.'
            f.write("%s\n" % msg_text)
            
    class_preds = []
    for message in messages:
        if not message:
            class_preds.append(np.zeros((1, N_CATEGORIES)))
            continue
        X = np.zeros((len(message), WORD_LENGTH, n_chars))
    
        for idx, word in enumerate(message):
            x = []
            for char in word[0:WORD_LENGTH].rjust(WORD_LENGTH):
                x.append(char_to_int[char])
            X[idx] = to_categorical(x, n_chars)

        pred = classification_model.predict(X)
        pred = sum(pred, 0) / pred.shape[0]
        class_preds.append(pred.tolist())
        
    with open(MESSAGE_PRED, 'wb') as fp:
        pickle.dump(class_preds, fp)
    
    X = []
    y = []
    for idx in range(len(messages) - 1):
        X.append(class_preds[idx])
        y.append(class_preds[idx+1] + to_categorical(char_to_int[messages[idx+1][0][0]], n_chars).tolist())

    X = np.array(X).astype(float)
    y = np.array(y).astype(float)
    
    model = Sequential()
    model.add(Dense(160, activation='relu'))
    model.add(BatchNormalization(scale=False))
    model.add(Dropout(0.3))
    model.add(Dense(160, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(y.shape[1], activation='relu'))
    opt = Adam(learning_rate=0.0001)
    model.compile(opt, 'mse', metrics=['accuracy'])
    
    model.fit(X,
              y,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True)
    
    model.save(RESPONSE_MODEL)

if __name__ == '__main__':
    args = parse_args()
    main()
