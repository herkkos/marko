import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

N_CATEGORIES = 3205
HISTORY_VAR = 30
PRED_LENGTH = 10
SPLITS = 500

C_FILE = '../categories_medium.txt'
PRED_FILE = '../pred_medium.txt'

RESPONSE_MODEL_NAME = 'medium/word_clf.h5'


def main():
    model = load_model(RESPONSE_MODEL_NAME)
    
    with open(C_FILE, 'r') as f:
        categories = f.readlines()
    
    split_size = len(categories) // SPLITS
    for split_idx in range(SPLITS):
        print(split_idx)
        split_categories = categories[split_idx*split_size : ((split_idx+1)*split_size) - 1]
        
        class_preds = []
        for line in split_categories:
            line_arr = []
            if line.strip():
                for x in line.strip().split(','):
                    line_arr.append(int(x))
                line_arr.append(N_CATEGORIES - 1)
            class_preds.append(line_arr)
        del split_categories
    
        XA = [] # Old words one-hot
        XB = [] # New words one-hot
        y = []
        prevs = []
        for pred_idx in range(0, len(class_preds) - 1):
            XA_msg = []
            XB_msg = []
            y_msg = []
            for cat in class_preds[pred_idx]:
                if cat != (N_CATEGORIES -1):
                    prevs.append(cat)
    
            if len(prevs) < HISTORY_VAR:
                continue
    
            prevs = prevs[-HISTORY_VAR:]
            xa = []
            for i in range(0, HISTORY_VAR):
                xa.append(to_categorical(prevs[i], N_CATEGORIES - 1))
    
            xb = np.array([[0] * (N_CATEGORIES - 1) + [1]] * PRED_LENGTH)
            for word_idx, word in enumerate(class_preds[pred_idx][:PRED_LENGTH]):
                yy = to_categorical(word, N_CATEGORIES)
                XA_msg.append(xa)
                XB_msg.append(xb)
                y_msg.append(yy)
                xb = np.vstack([xb, yy])[1:]
    
            XA.append(np.array(XA_msg, dtype=np.float32))
            XB.append(np.array(XB_msg, dtype=np.float32))
            y.append(np.array(y_msg, dtype=np.float32))
        
        preds = []
        for i in range(len(y)):
            msg_preds = []
            if len(XA[i]):
                score = model.predict([XA[i], XB[i]])
                for cat in score:
                    msg_preds.append(cat.argmax())
    
            preds.append(msg_preds)
            
        with open(PRED_FILE, 'a', newline='') as f:
            for pred in preds:
                f.write("%s\n" % ','.join(map(str, pred)))

if __name__ == '__main__':
    main()
