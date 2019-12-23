import json
import os

import gensim
from gensim import corpora, models
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

filepath = './input/sample_data.json'
data = {}
all_words = []

def readr(filepath):
    for line in open(filepath, mode="r"):
        cmnt = json.loads(line)
        tokens = word_tokenize(cmnt['body'])
        if len(tokens) > 3:
            stemmed = process(tokens)
            data[cmnt['id']] = {
                'id': cmnt['id'],
                'stemmed': stemmed,
                'subreddit': cmnt['subreddit']
            }
        pass


def process(tokens):
    result = []
    for word in tokens:
        word = word.lower()
        if word.isalpha() and word not in stop_words:
            stemmed = porter.stem(word)
            result.append(stemmed)
    all_words.append(result)
    return result


def prepare(data):
    if not os.path.isfile('./output/result.json'):
        readr(filepath)
        with open('./output/result.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
    else:
        with open('./output/result.json', 'r') as infile:
            data = json.load(infile)

    if not all_words:
        for v in data.values():
            lst = []
            for s in v['stemmed']:
                lst.append(s)
            all_words.append(lst)


if __name__ == '__main__':
    prepare(data)

    dictionary = gensim.corpora.Dictionary(all_words)

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in all_words]

    #tfidf = models.TfidfModel(bow_corpus)
    #corpus_tfidf = tfidf[bow_corpus]

    # Running LDA using Bag of Words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

