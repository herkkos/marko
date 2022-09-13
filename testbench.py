import random

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

N_CATEGORIES = 3205
HISTORY_VAR = 5
TRAIN_SPLITS = 50
CHARS_PRED_LENGTH = 160
WORDS_PRED_LENGTH = 15

X_FILE = 'X_data.json'
C_FILE = 'categories_medium.txt'

RESPONSE_MODEL_NAME = 'classification/marko_response_medium_2.h5'
GENERATE_MODEL_NAME = 'generator/marko_gene_medium_2.h5'

EMPTY_CHAR = '#'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö.' + EMPTY_CHAR
N_CHARS = len(CHARS)
CHAR_TO_INT = dict((w, i) for i, w in enumerate(CHARS))
INT_TO_CHAR = dict((i, w) for i, w in enumerate(CHARS))

N_TESTS = 50


def load_test_data(idx: int):
    with open(C_FILE, 'r') as f:
        categories = f.readlines()
        
    if not categories[idx].strip():
        return None
        
    split_categories = categories[idx - HISTORY_VAR : idx]
    del categories

    class_preds = []
    for line in split_categories:
        if line.strip():
            class_preds.append([int(x) for x in line.strip().split(',')])
        else:
            class_preds.append([])
    
    category_mat = []
    for prev_category_idx in reversed(range(1, HISTORY_VAR + 1)):
        classes_one_hot = np.zeros(N_CATEGORIES)
        for _class in class_preds[0-prev_category_idx]:
            classes_one_hot[_class] = 1
        category_mat.append(classes_one_hot)
        
    xx = np.array([[0] * (N_CATEGORIES - 1) + [1]] * WORDS_PRED_LENGTH)
        
    return (np.array(category_mat, dtype=np.float32), xx)

def predict_words(data):
    model = load_model(RESPONSE_MODEL_NAME)
    XA = np.expand_dims(data[0], 0)
    XB = np.expand_dims(data[1], 0)
    
    sentence_words = []
    while len(sentence_words) < WORDS_PRED_LENGTH:
        word = model.predict(x=[XA, XB])
        new_word = np.random.choice(N_CATEGORIES, p=np.squeeze(word, 0))
        if new_word == N_CATEGORIES - 1:
            return sentence_words
        else:
            sentence_words.append(new_word)
            
    return sentence_words

def predict_chars(words):
    model = load_model(GENERATE_MODEL_NAME)
    XA = np.zeros(N_CATEGORIES)
    for _class in words:
        XA[_class] = 1
    
    XB = np.array([[0] * (N_CHARS - 1) + [1]] * CHARS_PRED_LENGTH)

    XA = np.expand_dims(XA, 0)
    
    sentence_chars = []
    while len(sentence_chars) < CHARS_PRED_LENGTH:
        XB = np.expand_dims(XB, 0)
        char = model.predict(x=[XA, XB])
        new_char = INT_TO_CHAR[char.argmax()]
        if new_char == EMPTY_CHAR:
            return sentence_chars
        else:
            sentence_chars.append(new_char)
            XB = np.squeeze(XB, 0)
            XB = np.vstack([XB, np.expand_dims(to_categorical(char.argmax(), N_CHARS), 0)])[1:]
            
    return sentence_chars

def print_msg(chars):
    msg_str = ''
    for char in chars:
        msg_str += char
    print(msg_str)


def main():
    for test in range(N_TESTS):
        idx = random.randint(HISTORY_VAR, 199574)
        
        data = load_test_data(idx)
        if not data:
            continue
        words = predict_words(data)
        chars = predict_chars(words)
        print_msg(chars)
        
        

if __name__ == '__main__':
    main()
