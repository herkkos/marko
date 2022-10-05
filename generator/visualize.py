import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

NCLASSES = 13405
PRED_LENGTH = 160
HIST_LENGTH = 15

EMPTY_CHAR = '#'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö0123456789:❤👍😂😆😩🤣"()-:;,!?.' + EMPTY_CHAR
N_CHARS = len(CHARS)
CHAR_TO_INT = dict((c, i) for i, c in enumerate(CHARS))
INT_TO_CHAR = dict((i, c) for i, c in enumerate(CHARS))


BOW_FILE = '../bow_large.txt'
with open(BOW_FILE, 'r') as f:
    bow = f.readlines()
INT_TO_WORD = dict((i, w) for i, w in enumerate(bow))

model = load_model('large/word_gen.h5')

with open('../X_data.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
messages = [x['generation'] for x in data]

split_messages = messages[0 : 300]
del messages, data

with open('../categories_large.txt', 'r') as f:
    categories = f.readlines()
    
split_categories = categories[0 : 300]
del categories

class_preds = []
for line in split_categories:
    line_arr = []
    if line.strip():
        for x in line.strip().split(','):
            line_arr.append(int(x))
    class_preds.append(line_arr)
del split_categories


XA = [] # Categories for message
XB = [] # Previous characters of the message
y = []
for msg_idx, msg in enumerate(split_messages):
    msg_str = msg.strip() + EMPTY_CHAR
    
    xa = []
    for word in class_preds[msg_idx][:HIST_LENGTH]:
        xa.append(word)
    xa = [NCLASSES - 1] * HIST_LENGTH + xa
    xa = xa[-HIST_LENGTH:]
    
    for char_idx, char in enumerate(msg_str[:-1]):
        xx = []
        for i in reversed(range(char_idx + 1)):
            xx.append(CHAR_TO_INT[msg_str[char_idx-i]])
        xx = [CHAR_TO_INT[EMPTY_CHAR]] * PRED_LENGTH + xx
        xx = xx[-PRED_LENGTH:]
    
        XA.append(xa)
        XB.append(xx)
        y.append(to_categorical(CHAR_TO_INT[msg_str[char_idx + 1]], N_CHARS))

XA = np.array(XA).astype(float)
XB = np.array(XB).astype(float)
y = np.array(y).astype(float)

resp_str = ''
for i in range(0, 100):
    XA_test = np.expand_dims(XA[i], 0)
    XB_test = np.expand_dims(XB[i], 0)
    score = model.predict([XA_test, XB_test])
    resp_str += INT_TO_CHAR[score.argmax()]
    print(INT_TO_CHAR[y[i].argmax()] + " : " + INT_TO_CHAR[score.argmax()])
