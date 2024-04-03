import json
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

X_FILE = '../X_data.json'
BERT_FILE = '../all_msgs_knowledge.txt'
CATEGORY_FILE = '../categories_bert_knowledge.txt'
BERT_NAME = 'bert-marko-knowledge'
VOCAB_FILE = '../bert-marko-knowledge-vocab.txt'

with open(X_FILE, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

text_ds = ''
for msg in data:
    if msg['generation'] != '':
        text_ds += msg['generation'] + '. '

with open(BERT_FILE, 'w', encoding='utf-8') as f:
    f.write(text_ds)

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=None,
    lowercase=True
)

tokenizer.train(files=BERT_FILE, vocab_size=10000, min_frequency=5,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

tokenizer.save_model('.', BERT_NAME)

tokenizer = BertTokenizer.from_pretrained(VOCAB_FILE)

categories = []
msg_text = ''
prev_sender = data[0]['sender']
for _, message in enumerate(data):
    msg_text = message['generation']
    word_categories = tokenizer(msg_text)['input_ids']
    word_categories.insert(0, prev_sender)
    word_categories.insert(1, message['date'])
    categories.append(word_categories)

with open(CATEGORY_FILE, 'w', newline='') as f:
    for cats in categories:
        f.write("%s\n" % ','.join(str(x) for x in cats))
