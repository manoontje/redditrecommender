import json

import gensim
import nltk
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

filepath = './sample_data.json'
d = {}
all_words = []

# moet een map functie worden die een preprocess aanroept
def readr(filepath):
    cnt = 1
    for line in open(filepath, mode="r"):
        cmnt = json.loads(line)
        tokens = word_tokenize(cmnt['body'])
        if len(tokens) > 3:
            stemmed = preprocess(tokens)
            #print('{} : {}'.format(cnt, stemmed))
            cnt += 1
            yield { 'id'        : cmnt['id'],
                    'stemmed'   : stemmed,
                    'subreddit' : cmnt['subreddit'] }
        pass

def preprocess(tokens):
    result = []
    for word in tokens:
        word = word.lower()
        if word.isalpha() and word not in stop_words:
            stemmed = porter.stem(word)
            result.append(stemmed)
    all_words.append(result)
    return result

if __name__ == '__main__':
    lst = list(readr(filepath))
    #print(all_words)
    dictionary = gensim.corpora.Dictionary(all_words)
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in all_words]
    bow_corpus[28]