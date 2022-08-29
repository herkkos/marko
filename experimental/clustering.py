import csv
import json
from Levenshtein import distance as lev
from matplotlib import pyplot as plt
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans



RAW_FILE = 'result.json'
BOW_FILE = 'bow_auto.txt'
C_FILE = 'categories_auto.csv'
X_FILE = 'X_data.json'
MAX_LENGTH = 160
MIN_COUNT = 2
MAX_COUNT = 1000
MIN_LENGTH = 2
MAX_DIFFERENCE = 2

CHARS = ' abcdefghijklmnopqrstuvwxyzåäö'

def contains_number(s):
    return any(i.isdigit() for i in s)

def main():
    with open(RAW_FILE, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    bow = []
    bow_count = []
    X = []
    X_str = ''
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
            X.append(formated_msg)
            for word in formated_msg.split():
                if word not in bow:
                    bow.append(word.strip())
                    bow_count.append(1)
                else:
                    bow_count[bow.index(word)] += 1
    X_str = ' '.join(X)
    
    vectorizer_cv = CountVectorizer(analyzer='word')
    X_cv = vectorizer_cv.fit_transform(X)
    vectorizer_wtf = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
    X_wtf = vectorizer_wtf.fit_transform(X)
    
    lda = LatentDirichletAllocation(n_components=30, learning_decay=0.9)
    X_lda = lda.fit(X_cv)

    sse={}
    for k in np.arange(100,900,100):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_cv)
        sse[k] = kmeans.inertia_
        
    plt.plot(list(sse.keys()),list(sse.values()))
    plt.xlabel('Values for K')
    plt.ylabel('SSE')
    plt.show()
    
    der_vals = []
    for i in range(7):
        der_vals.append(abs(list(sse.values())[i+1] - list(sse.values())[i]))
    
    plt.plot(list(sse.keys())[1:], der_vals)
    plt.xlabel('Values for K')
    plt.ylabel('SSE')
    plt.show()
    
    kmeans = KMeans(n_clusters=500)
    kmeans.fit(X_cv)
    result = pd.concat([X ,pd.DataFrame(X_cv.toarray(),columns=vectorizer_cv.get_feature_names())],axis=1)
    result['cluster'] = kmeans.predict(X_cv)
            
if __name__ == '__main__':
    main()
