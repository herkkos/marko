import csv
import json
import numpy as np


RAW_FILE = 'result.json'
BOW_FILE = 'bow1000.txt'
C_FILE = 'categories1000.csv'
X_FILE = 'X_data.json'
MAX_LENGTH = 160
MIN_COUNT = 14
MAX_COUNT = 101

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'

def contains_number(s):
    return any(i.isdigit() for i in s)

def main():
    with open(RAW_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    bow = []
    bow_count = []
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
                if word not in bow:
                    bow.append(word.strip())
                    bow_count.append(1)
                else:
                    bow_count[bow.index(word)] += 1

    new_bow = []
    new_count = []
    for word, count in zip(bow, bow_count):
        if MIN_COUNT < count < MAX_COUNT and len(word) > 2 and not contains_number(word):
            new_bow.append(word)
            new_count.append(count)

    bow = new_bow

    with open(BOW_FILE, 'w') as f:
        for word in bow:
            f.write("%s\n" % word)
            
    with open(X_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    categories = np.zeros((len(data), len(bow)))
    for idx, message in enumerate(data):
        msg_text = message['formated'].split()
        if len(message['text']) > MAX_LENGTH:
            continue
        for w_idx, word in enumerate(bow):
            for msg_word in msg_text:
                if word in msg_word:
                    categories[idx][w_idx] = 1
    categories = categories.astype(int)

    with open(C_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(categories)

if __name__ == '__main__':
    main()
