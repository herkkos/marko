import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from utils import parse_args

REUSES = 2
BATCH_SIZE = 128
EPOCHS = 50
N_SPLITS = 10
N_CATEGORIES = 1046
MAX_LENGTH = 160
PRED_LENGTH = 15

C_FILE = 'categories1000.csv'
X_FILE = 'X_data.json'
PRED_FILE = 'class_preds1000'
MODEL_NAME = 'marko_gen_1000_15.h5'

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö.'

def main():
    with open(PRED_FILE, 'rb') as fp:
        preds = pickle.load(fp)
    
    # with open(X_FILE, 'r', encoding='utf-8') as json_file:
    #     data = json.load(json_file)
    with open('X.txt', 'r') as fp:
        data = fp.readlines()
        
    chars = CHARS
    n_chars = len(chars)
    
    char_to_int = dict((w, i) for i, w in enumerate(chars))
    int_to_char = dict((i, w) for i, w in enumerate(chars))
    
    
    
### KAKSI INPUTTIA
    # Input
    # inputA = Input(shape=(N_CATEGORIES,))
    # inputB = Input(shape=(PRED_LENGTH, n_chars))
    # # Categories
    # x = Dense(N_CATEGORIES, activation='relu')(inputA)
    # x = BatchNormalization()(x)
    # x = Dense(100, activation='relu')(x)
    # x = Model(inputs=inputA, outputs=x)
    # # Chars
    # y = LSTM(PRED_LENGTH * n_chars + 1, return_sequences=True)(inputB)
    # y = BatchNormalization()(y)
    # y = Dropout(0.33)(y)
    # y = LSTM(PRED_LENGTH * n_chars + 1, return_sequences=True)(inputB)
    # y = BatchNormalization()(y)
    # y = Dropout(0.33)(y) 
    # y = LSTM(PRED_LENGTH * n_chars + 1, return_sequences=False)(inputB)
    # y = BatchNormalization()(y)
    # y = Dropout(0.33)(y)
    # y = Model(inputB, outputs=y)
    # # Combine
    # combined = Concatenate()([x.output, y.output])    
    # z = Dense(n_chars, activation='relu')(combined)
    # z = Dense(n_chars, activation='relu')(z)
    # # Model
    # model = Model(inputs=[x.input, y.input], outputs=z)
    # opt = Adam(learning_rate=0.0001)
    # model.compile(opt, 'mse', metrics=['accuracy'])
    
    
    model = load_model(MODEL_NAME)

    
    split_size = len(data) // N_SPLITS
    for j in range(REUSES):
        for i_s in range(N_SPLITS):
            print(f"Training iteration {j+1} with split {i_s+1}")
            XA = []
            XB = []
            y = []
            for msg_idx, msg in enumerate(data[i_s*split_size:((i_s+1)*split_size) - 1]):
                msg_str = ' '*PRED_LENGTH + msg.strip()
                for idx, char in enumerate(msg_str[:-1]):
                    if idx < PRED_LENGTH:
                        continue

                    xx = []
                    for i in reversed(range(PRED_LENGTH)):
                        xx.append(to_categorical(char_to_int[msg_str[idx-i]], n_chars))
                        # xx.append(msg_str[idx-i])
                    
                    if len(preds[i_s*split_size + msg_idx]) != N_CATEGORIES:
                        continue
                    XA.append(preds[i_s*split_size + msg_idx])
                    XB.append(xx)
                    y.append(to_categorical(char_to_int[msg_str[idx+1]], n_chars))
                    # y.append(msg_str[idx+1])
                    
            XA = np.array(XA).astype(float)
            XB = np.array(XB).astype(float)
            y = np.array(y).astype(float)
            
            model.fit(x=[XA, XB],
                      y=y,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      shuffle=True)
    # for j in range(REUSES):
    #     for i_s in range(N_SPLITS):
    #         print(f"Training iteration {j+1} with split {i_s+1}")
    #         XA = []
    #         XB = []
    #         y = []
    #         for msg_idx, msg in enumerate(data[i_s*split_size:((i_s+1)*split_size) - 1]):
    #             if msg['formated'] == '':
    #                 continue
    #             msg_str = ' '*PRED_LENGTH + msg['formated'] + '.'
    #             for idx, char in enumerate(msg_str[:-1]):
    #                 if idx < PRED_LENGTH:
    #                     continue

    #                 xx = []
    #                 for i in reversed(range(PRED_LENGTH)):
    #                     xx.append(to_categorical(char_to_int[msg_str[idx-i]], n_chars))
    #                     # xx.append(msg_str[idx-i])
                    
    #                 if len(preds[i_s*split_size + msg_idx]) != N_CATEGORIES:
    #                     continue
    #                 XA.append(preds[i_s*split_size + msg_idx])
    #                 XB.append(xx)
    #                 y.append(to_categorical(char_to_int[msg_str[idx+1]], n_chars))
    #                 # y.append(msg_str[idx+1])
                    
    #         XA = np.array(XA).astype(float)
    #         XB = np.array(XB).astype(float)
    #         y = np.array(y).astype(float)
            
    #         model.fit(x=[XA, XB],
    #                   y=y,
    #                   epochs=EPOCHS,
    #                   batch_size=BATCH_SIZE,
    #                   shuffle=True)
        
    model.save(MODEL_NAME)
    
    
    
### YKSI INPUT
            
    # gen_model = Sequential()
    # gen_model.add(LSTM(160, input_shape=(PRED_LENGTH, preds[0].shape[1] + n_chars), return_sequences=True))
    # gen_model.add(BatchNormalization())
    # gen_model.add(Dropout(0.3))
    # gen_model.add(LSTM(160, return_sequences=True))
    # gen_model.add(BatchNormalization())
    # gen_model.add(Dropout(0.3))
    # gen_model.add(LSTM(160))
    # gen_model.add(BatchNormalization())
    # gen_model.add(Dropout(0.3))
    # gen_model.add(Dense(160, activation='relu'))
    # gen_model.add(BatchNormalization())
    # gen_model.add(Dense(n_chars, activation='relu'))
    # opt = Adam(learning_rate=0.00001)
    # gen_model.compile(opt, 'mse', metrics=['accuracy'])
            
    # split_size = len(data) // N_SPLITS
    # for j in range(REUSES):
    #     for i_s in range(N_SPLITS):
    #         print(f"Training iteration {j+1} with split {i_s+1}")
    #         X = []
    #         y = []
    #         for msg_idx, msg in enumerate(data[i_s*split_size:((i_s+1)*split_size) - 1]):
    #             if msg['formated'] == '':
    #                 continue
    #             msg_str = ' '*PRED_LENGTH + msg['formated'] + '.'
    #             for idx, char in enumerate(msg_str[:-1]):
    #                 if idx < PRED_LENGTH:
    #                     continue
    #                 yy = []
    #                 for i in reversed(range(PRED_LENGTH)):
    #                     char_y = np.array(preds[i_s*split_size + msg_idx])
    #                     char_y = char_y / max(char_y)
    #                     char_y = np.append(char_y, to_categorical(char_to_int[msg_str[idx-i]], n_chars))
    #                     yy.append(char_y)
    #                 y.append(yy)
    #                 X.append(to_categorical(char_to_int[msg_str[idx+1]], n_chars))
                    
    #         X = np.array(X).astype(float)
    #         y = np.array(y).astype(float)
            
    #         for idx, char in enumerate(CHARS):
    #             print(f"{char} count: {np.count_nonzero(X[:,idx])}")
            
    #         gen_model.fit(y,
    #                       X,
    #                       epochs=EPOCHS,
    #                       batch_size=BATCH_SIZE,
    #                       shuffle=True)
            
    # gen_model.save(MODEL_NAME)


if __name__ == '__main__':
    args = parse_args()
    main()
