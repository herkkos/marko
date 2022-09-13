import json
from math import floor
from statistics import median

from Levenshtein import distance as lev
import numpy as np


RAW_FILE = 'result.json'
BOW_FILE = 'bow_1000.txt'
COUNT_FILE = 'bow_count.txt'
CORR_FILE = 'corr_median.txt'
CATEGORY_FILE = 'categories_median.txt'
X_FILE = 'X_data.json'
MIN_COUNT = 50
MAX_COUNT = 750
MIN_LENGTH = 2
LEV_FACTOR = 0.35

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'
GEN_CHARS = ' abcdefghijklmnopqrstuvwxyzåäö0123456789:❤👍😂😆😩🤣"()-:;,!?.'

def contains_number(s):
    return any(i.isdigit() for i in s)

def main():
    with open(RAW_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    bow = []
    bow_count = []
    X = []
    chats = data['chats']
    for chat in chats['list']:
        for msg in chat['messages']:
            if 'from' not in msg:
                continue
            msg_text = msg['text']

            if type(msg_text) != str:
                continue

            gen_msg = ''
            formated_msg = ''
            for char in msg_text.lower():
                if char in CHARS:
                    formated_msg += char
                    gen_msg += char
                elif char in GEN_CHARS:
                    gen_msg += char
            for word in formated_msg.split():
                for b_word in bow:
                    if lev(word, b_word) <= floor(len(word)*LEV_FACTOR):
                        bow_count[bow.index(b_word)] += 1
                        break
                else:
                    bow.append(word.strip())
                    bow_count.append(1)
                    
            msg = {'date': msg['date'],
                   'text': msg_text,
                   'formated': formated_msg,
                   'generation': gen_msg,
                   'sender': msg['from']}
            X.append(msg)

    new_bow = []
    new_count = []
    non_used = []
    for word, count in zip(bow, bow_count):
        if MIN_COUNT < count and MAX_COUNT > count and len(word) >= MIN_LENGTH and not contains_number(word):
            new_bow.append(word)
            new_count.append(count)
        else:
            non_used.append(word)
            
    bow, bow_count = zip(*sorted(zip(new_bow, new_count)))
    
    with open(X_FILE, 'w') as f:
        json.dump(X, f)

    with open(BOW_FILE, 'w') as f:
        for word in bow:
            f.write("%s\n" % word)
            
    # with open(COUNT_FILE, 'w') as f:
    #     for word in bow_count:
    #         f.write("%s\n" % word)
            
            
            
    with open(BOW_FILE, 'r') as f:
        bow = f.readlines()
        
    with open(X_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    
    categories = []
    for idx, message in enumerate(data):
        msg_text = message['formated'].split()
        word_categories = []
        for msg_word in msg_text:
            for w_idx, word in enumerate(bow):
                if lev(word, msg_word) <= floor(len(word)*LEV_FACTOR):
                    word_categories.append(str(w_idx))
                    break
        categories.append(word_categories)
        
    class_preds = []
    for line in categories:
        if len(line) > 0:
            # for x in line[0].strip().split(','):
            for x in line:
                class_preds.append(int(x))
            class_preds.append(len(bow))

    class_preds = np.array(class_preds)
    counts = []
    for word in range(len(bow)):
        counts.append(np.count_nonzero(class_preds == word))

    max_count = max(counts)
    median_count = median(counts)
    fix_factor = median_count / max_count
    corr_factors = []
    for word in range(len(bow) + 1):
        corr_factors.append((np.count_nonzero(class_preds[class_preds == word]) / max_count ) - fix_factor)
    
    with open(CORR_FILE, 'w') as f:
        for word in corr_factors:
            f.write("%s\n" % word)

    with open(CATEGORY_FILE, 'w', newline='') as f:
        for cats in categories:
            f.write("%s\n" % ','.join(cats))

        
if __name__ == '__main__':
    main()
    