import json
from math import floor

from Levenshtein import distance as lev


RAW_FILE = 'result.json'
BOW_FILE = 'bow_medium.txt'
C_FOLDER = 'categories_medium'
X_FILE = 'X_data.json'
MAX_LENGTH = 160
MIN_COUNT = 20
MIN_LENGTH = 2
LEV_FACTOR = 0.35
CATEGORY_BATCH_SIZE = 5000

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'

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

            formated_msg = ''
            for char in msg_text.lower():
                if char in CHARS:
                    formated_msg += char
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
                   'sender': msg['from']}
            X.append(msg)

    new_bow = []
    new_count = []
    non_used = []
    for word, count in zip(bow, bow_count):
        if MIN_COUNT < count and len(word) >= MIN_LENGTH and not contains_number(word):
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

    with open('categories_medium.txt', 'w', newline='') as f:
        for cats in categories:
            f.write("%s\n" % ','.join(cats))

        
if __name__ == '__main__':
    main()
    