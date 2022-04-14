import json
from utils import parse_args

CHAT_FILE = 'result.json'
OUTPUT_FILE = 'X_data.json'
BOW_FILE = 'bow.txt'
EMPTY = ['']
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'

bow = []
bow_count = []
X = []


def main():
    with open(CHAT_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    chats = data['chats']
    for chat in chats['list']:
        for msg in chat['messages']:
            if 'from' not in msg:
                continue
            msg_time = msg['date']
            msg_text = msg['text']
            msg_sender = msg['from']
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

            msg = {'date': msg_time,
                   'text': msg_text,
                   'formated': formated_msg,
                   'sender': msg_sender}
            X.append(msg)


    with open(OUTPUT_FILE, 'w') as f:
        json.dump(X, f)

    with open(BOW_FILE, 'w') as f:
        for word in bow:
            f.write("%s\n" % word)


if __name__ == '__main__':
    args = parse_args()
    main()
