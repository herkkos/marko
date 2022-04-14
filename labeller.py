import json

FILE = 'X_data.json'

def main():
    with open('categories.txt', 'r') as file:
        categories = file.readlines()
    
    with open('emotions.txt', 'r') as file:
        emotions = file.readlines()
        
    num_lines = sum(1 for line in open('categories.txt'))
    
    with open(FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        for idx, message in enumerate(data):
            if idx < num_lines:
                continue
            msg_text = message['text']
            msg_sender = message['sender']
            msg_date = message['date']
            print(f"{msg_date}\t{msg_sender}\t{msg_text}")
            category = input("0: EMPTY, 1: VIINA, 2: KANNABIS, 3: ÖRINÄ, 4: PAKOTUKSET")
            emotion = input("0: EMPTY, 1: IRONINEN, 2: ILOINEN, 3: VIHAINEN")
            if category == 'q' or emotion == 'q':
                break
            categories.append(category)
            emotions.append(emotion)

    with open('categories.txt', 'w') as f:
        for item in categories:
            f.write("%s\n" % item)

    with open('emotions.txt', 'w') as f:
        for item in emotions:
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()
