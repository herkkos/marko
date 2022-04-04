import json


EMPTY = ['']
# CHARS = ' abcdefghijklmnopqrstuvwxyzåäö.:,;()_-?!&'
CHARS = ' abcdefghijklmnopqrstuvwxyzåäö.'
MIN_COUNT = 2

bow = []
bow_count = []
y = []
X = []


def main():
    with open('result.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        chats = data['chats']
        for chat in chats['list']:
            prev_msg = ''
            curr_msg = ''
            prev_sender = ''
            for msg in chat['messages']:
                msg_text = msg['text']
                if 'from' not in msg:
                    continue
                msg_sender = msg['from']
                if type(msg_text) != str:
                    continue
                msg_text = ''.join(c for c in msg_text.lower() if c in CHARS)
                if msg_text in EMPTY:
                    continue
                for word in msg_text.split():
                    if word not in bow:
                        bow.append(word.strip())
                        bow_count.append(1)
                    else:
                        bow_count[bow.index(word)] += 1
                if msg_sender == prev_sender or prev_sender == '':
                    curr_msg = curr_msg + ' ' + msg_text + ' .'
                    prev_sender = msg_sender
                else:
                    X.append(prev_msg.lstrip() + ' .')
                    y.append(curr_msg.lstrip() + ' .')
                    prev_msg = curr_msg
                    curr_msg = msg_text
                    prev_sender = msg_sender

    clean_bow = []
    for word, count in zip(bow, bow_count):
        if count > MIN_COUNT:
            clean_bow.append(word)

    with open('X.txt', 'w') as f:
        for word in X:
            f.write("%s\n" % word)

    with open('y.txt', 'w') as f:
        for word in y:
            f.write("%s\n" % word)
            
    with open('bow.txt', 'w') as f:
        for word in clean_bow:
            f.write("%s\n" % word)


if __name__ == '__main__':
    main()
