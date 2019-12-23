import json
import datetime
import os.path
from os import path

import gensim
from gensim import corpora, models
from gensim.test.utils import get_tmpfile

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

stop_words = set(stopwords.words('english'))
# Nonsense words have the same purpose as stop words; after stemming
nonse_words = ['http', 'https']
stemmer = SnowballStemmer("english")

MAX_ITER = 1000000
NUM_TOPICS = 20
SUBREDDITS = []

def filter_comments(filepath, dictionary, words):
    '''(1) Filters fields 'id', 'body' and 'subreddit' of JSON objects and sets as new dictionary entries;
    (2) new 'body' field is tokenized en lemmatized and
    (3) adds all newly stemmed tokens to words list.'''
    count = 0
    for line in open(filepath, 'r'):
        if count > MAX_ITER:
            break
        JSON_object = json.loads(line)
        # Extract fields of interest, being 'id', 'body' and 'subreddit'.
        ID        = JSON_object['id']
        BODY      = JSON_object['body']
        SUBREDDIT = JSON_object['subreddit']
        # Tokenize and lemmatize raw BODY entry.
        TOKENS = process(BODY)
        # Add all tokens to all_words.
        words.append(TOKENS)
        # Add SUBREDDIT to SUBREDDITS to count total /r/
        SUBREDDITS.append(SUBREDDIT)
        # Set newly filtered entries into dictionary, replacing BODY for stemmed counterpart.
        dictionary[ID] = { 'id' : ID, 'body' : TOKENS, 'subreddit': SUBREDDIT }
        count += 1

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def process(text):
    '''Tokenize and lemmatize.'''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        # If token is not a stop word and longer than three characters. Statement: `len(token) > 3` especially helpful
        # to filter out alphabetical, but nonsensical entries. (NOT lol, www, com, nl etc.)
        if token not in stop_words and len(token) > 3:
            # Stem and add to result
            stemmed_token = lemmatize_stemming(token)
            if stemmed_token not in nonse_words:
                result.append(stemmed_token)
    return result


if __name__ == '__main__':
    if not path.exists('./lda.model'):
        # Path to file containing JSON objects.
        #infile = './sample_data.json'
        infile = './RC_2017-12'
        # Dictionary to save said filtered JSON objects.
        data = dict()
        # List containing (multiple of) all tokenized and stemmed words.
        all_words = list()

        print('Timstamp: {}\t Start'.format(datetime.datetime.now()))
        filter_comments(infile, data, all_words)
        print('Timestamp: {}\t Finish'.format(datetime.datetime.now()))

        dictionary = corpora.Dictionary(all_words)

        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        dictionary.save_as_text('./dictionary')

        bow_corpus = [dictionary.doc2bow(doc) for doc in all_words]

        #tfidf = models.TfidfModel(bow_corpus)
        #corpus_tfidf = tfidf[bow_corpus]

        # Running LDA using Bag of Words.
        lda_model = models.LdaMulticore(bow_corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=2, workers=2)
        # Save model to disk
        lda_model.save('lda.model')
    else:
        lda_model = models.LdaModel.load('./lda.model')
        dictionary = corpora.Dictionary.load_from_text('./dictionary')

    #for idx, topic in lda_model.print_topics(-1):
    #    print('Topic: {} \nWords: {}'.format(idx, topic))

    # Prompt the user to paste a comment he/she liked.
    _input = input('Prompt: ')
    # Recommend Topic based on previous user input.
    bow_vector = dictionary.doc2bow(process(_input))

    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
        print("Score: {}\t Index: {}\tTopic: [{}]".format(score, index, lda_model.print_topic(index, 6)))

    #print('Total subreddits: {}'.format(len(set(SUBREDDITS))))
