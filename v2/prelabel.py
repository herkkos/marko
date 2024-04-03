import argparse
import json

RAW_FILE = 'result.json'
X_FILE = 'combined_results.json'


CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'
GEN_CHARS = r""" abcdefghijklmnopqrstuvwxyzåäö0123456789❤👍😂😆😩🤣😊😋🤢😅🤝🔥⁉️💀🥴😭🤔😟😔🥰😍🥺🖕👌💅🏻🌚💸#&/=€@+*<>"()'-_:;,!?."""


def main(args):
    with open(RAW_FILE, 'r', encoding='utf-8') as json_file:
    # with open(args.json, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    X = []
    chats = data['chats']
    for chat in chats['list']:
        
        if 'name' not in chat:
            continue
        
        if chat['name'] != 'Elli':
            continue
        
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
                    
            msg = {'date': msg['date'],
                   'text': msg_text,
                   'formated': formated_msg,
                   'generation': gen_msg,
                   'sender': msg['from']}
            X.append(msg)

    with open(X_FILE, 'w') as f:
        json.dump(X, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prelabel messages.")
    parser.add_argument("--json", type=str, help="Message file")
    args = parser.parse_args()
    main(args)