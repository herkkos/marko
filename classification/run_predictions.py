import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

N_CATEGORIES = 1479
HISTORY_VAR = 20
PRED_LENGTH = 10
SPLITS = 100

C_FILE = '../categories_1000.txt'
PRED_FILE = '../pred_1000.txt'

RESPONSE_MODEL_NAME = 'median1000/word_clf.h5'


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
    

        prevs = []
        preds = []
        for pred_idx in range(0, len(class_preds) - 1):
            msg_pred = []
            for cat in class_preds[pred_idx]:
                if cat != (N_CATEGORIES -1):
                    prevs.append(cat)
    
            if len(prevs) < HISTORY_VAR:
                preds.append([])
                continue
    
            prevs = prevs[-HISTORY_VAR:]
            xa = []
            for i in range(0, HISTORY_VAR):
                xa.append(to_categorical(prevs[i], N_CATEGORIES - 1))
            xa = np.array(xa)
    
            xb = np.array([[0] * (N_CATEGORIES - 1) + [1]] * PRED_LENGTH)
            for idx in range(PRED_LENGTH):
                te_xa = np.expand_dims(xa, 0)
                te_xb = np.expand_dims(xb, 0)
                pred = model.predict(x=[te_xa, te_xb])
                msg_pred.append(pred.argmax())
                yy = to_categorical(pred.argmax(), N_CATEGORIES)
                xb = np.vstack([xb, yy])[1:]
            
            preds.append(msg_pred)
            
            
        with open(PRED_FILE, 'a', newline='') as f:
            for pred in preds:
                f.write("%s\n" % ','.join(map(str, pred)))

if __name__ == '__main__':
    main()
