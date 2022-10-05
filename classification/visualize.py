import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

NCLASSES = 1480
# NCLASSES = 13406
PRED_LENGTH = 10
HIST_LENGTH = 20

BOW_FILE = '../bow_1000.txt'
# BOW_FILE = '../bow_large.txt'
with open(BOW_FILE, 'r') as f:
    bow = f.readlines()
INT_TO_WORD = dict((i, w) for i, w in enumerate(bow))

model = load_model('median_new/word_clf.h5')
# model = load_model('large/word_clf.h5')

with open('../categories_1000.txt', 'r') as f:
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


XA = [] # Old words one-hot
XB = [] # New words one-hot
y = []
for pred_idx in range(0, len(class_preds)):
    if pred_idx < HIST_LENGTH:
        continue

    xa = []
    for i in reversed(range(1, HIST_LENGTH)):
        for cat in class_preds[pred_idx-i]:
            xa.append(cat)
    xa = [NCLASSES - 1] * HIST_LENGTH + xa
    xa = xa[-HIST_LENGTH:]

    for word_idx, word in enumerate(class_preds[pred_idx][:PRED_LENGTH]):
        yy = to_categorical(word, NCLASSES)
        
        xb = []
        for i in reversed(range(word_idx)):
            xb.append(class_preds[pred_idx][word_idx-i])
        xb = [NCLASSES - 1] * PRED_LENGTH + xb
        xb = xb[-PRED_LENGTH:]
        
        XA.append(xa)
        XB.append(xb)
        y.append(yy)

XA = np.array(XA, dtype=np.float32)
XB = np.array(XB, dtype=np.float32)
y = np.array(y, dtype=np.float32)

for i in range(200, 250):
    XA_test = np.expand_dims(XA[i], 0)
    resp_str = ''
    XB_test = np.expand_dims(XB[i], 0)
    score = model.predict([XA_test, XB_test])
    print(INT_TO_WORD[y[i].argmax()] + " : " + INT_TO_WORD[score.argmax()])
    # real_y = int(y[i])
    # print(INT_TO_WORD[real_y].strip() + " : " + INT_TO_WORD[score.argmax()])
