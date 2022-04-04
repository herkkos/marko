from Levenshtein import distance as lev
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Bidirectional


def main():
    raw_text = open('X.txt', 'r').read()
    with open('bow.txt', 'r') as file:
        bow = [line.strip() for line in file]
    bow.sort()
    
    clean_bow = ['.']
    for idx, word in enumerate(bow):
        if '.' in word or 'å' in word:
            continue
        if bow[idx-1] in word:
            continue
        clean_bow.append(word)
    bow = clean_bow
    
    word_to_int = dict((w, i) for i, w in enumerate(bow))
    int_to_word = dict((i, w) for i, w in enumerate(bow))
 
    n_vocab = len(bow)
    
    seq_length = 10
    dataX = []
    dataY = []
    
    lines = raw_text.splitlines()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        seq_in = lines[idx-1]
        seq_in = seq_in.split()[0:seq_length]
        while seq_in[len(seq_in)-1] == '.' and seq_in[len(seq_in)-2] == '.' and len(seq_in) > 1:
            seq_in.pop()
        
        for _ in range(seq_length - len(seq_in)):
            seq_in[0:0] = ['.']
        
        seq_out = line.split()[0:seq_length]
        while seq_out[len(seq_out)-1] == '.' and seq_out[len(seq_out)-2] == '.' and len(seq_out) > 1:
            seq_out.pop()
            
        for _ in range(seq_length - len(seq_out)):
            seq_out[0:0] = ['.']
        
        dataX.append([word_to_int[min(word_to_int.keys(), key=lambda x: lev(word, x))] for word in seq_in])
        dataY.append([word_to_int[min(word_to_int.keys(), key=lambda x: lev(word, x))] for word in seq_out])
        # dataX.append([word_to_int[word] for word in seq_in])
        # dataY.append([word_to_int[word] for word in seq_out])

    n_patterns = len(dataX)
    print("Total Patterns: "+ str(n_patterns))
    
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = np.reshape(dataY, (n_patterns, seq_length, 1))
    y = y / float(n_vocab)

    #TODO: pienemmällä sanakirjalla tämän käyttö on mahdollinen
    # X = to_categorical(dataX)
    # y = to_categorical(dataY)

    inputs = Input(shape=(seq_length, 1))
    encoder = LSTM(256, return_sequences=False)(inputs)
    encoding_repeat = RepeatVector(seq_length)(encoder)
    decoder = LSTM(256, return_sequences=True)(encoding_repeat)
    dense1 = Dense(160, activation='relu')(decoder)
    sequence_prediction = TimeDistributed(Dense(1, activation='relu'))(dense1)
    model = Model(inputs, sequence_prediction)
    model.compile('adam', 'mse')
    model.fit(X, y, epochs=150, batch_size=128)
    
    
    # model = Sequential()
    # model.add(Bidirectional(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(256)))
    # model.add(Dropout(0.2))
    # model.add(Dense(y.shape[1], activation='relu'))
    # model.add(Dense(y.shape[1], activation='relu'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X, y, epochs=150, batch_size=128)
    
    
    test_str = "jos marko lehtinen olisi hyvä työnantaja".split()
    for _ in range(seq_length - len(test_str)):
        test_str[0:0] = ['.']
    test_str = ([word_to_int[char.lower()] for char in test_str])
    test_y = np.array(test_str) / float(n_vocab)

    test_y = np.expand_dims(test_y, 1)
    test_y = np.expand_dims(test_y, 0)

    pred = model.predict(test_y)
    pred = pred.squeeze(0)
    pred = pred.squeeze(1)
    
    pred = pred * float(n_vocab)
    pred = [int_to_word[round(elem)] for elem in pred.tolist()]
    
    print(' '.join(pred))

