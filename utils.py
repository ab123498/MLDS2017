import pickle
import collections
import numpy as np
import math
import gensim
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer

class DataLoader():
    def __init__(self, voca_size, seq_length):
        self.voca_size = voca_size
        self.seq_length = seq_length
        
        self.tokenizer = RegexpTokenizer(r'[\w#]+\'?[\w#]+')
        
        training_data = open('data/corpus.txt', 'r').read().lower().split('\n')
        self.training_data = self._tokenize(training_data)
        
        self.vocabulary, self.vocabulary_inv = self._build_voca()
        
    def _tokenize(self, data):
        tokenizer = self.tokenizer
        data = [tokenizer.tokenize(sent) for sent in data]
        data = [s for s in data if len(s) > 0]
        
        return data
    
    def _preprocess(self, data, seq_length):
        vocabulary = self.vocabulary
        
        data_id = np.zeros((len(data), seq_length)).astype(int)
        for i, sent in enumerate(data):
            for j, word in enumerate(sent[:seq_length]):
                if word in vocabulary:
                    data_id[i][j] = vocabulary[word]
                else:
                    data_id[i][j] = vocabulary['other']
        return data_id

    
    def _build_voca(self):
        data = self.training_data
        words = []
        for sent in data:
            words += sent
        counter = collections.Counter(words)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in counter.most_common(self.voca_size-2)]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        
        # i start from 1 spare zero for padding
        vocabulary = {x: i+2 for i, x in enumerate(vocabulary_inv)}
        vocabulary['NULL_TAG'] = 0
        vocabulary['_____'] = 1
        return vocabulary, vocabulary_inv
    
    def train_batch(self, batch_size):
        data = self.training_data
        vocabulary = self.vocabulary
        data_len = len(data)

        x = np.zeros((len(data), self.seq_length)).astype(int)
        y = np.zeros((len(data))).astype(int)

        for i, sent in enumerate(data):
            for j, word in enumerate(sent[:self.seq_length]):
                if not word in vocabulary:
                    x[i][j] = vocabulary['other']
                else:
                    x[i][j] = vocabulary[word]
            l = len(sent)
            if l > self.seq_length:
                l = self.seq_length
            try:
                place2fill = np.random.randint(math.floor(l/4), math.ceil(l/4*3))
            except:
                print(l)
            y[i] = x[i][place2fill]
            x[i][place2fill] = 1
        
        for i in range(math.ceil(data_len/batch_size)):
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

    def test_batch(self, batch_size):
        import pandas as pd
        df = pd.read_csv('hw1_data/testing_data.csv')
        df.question = df.question.str.lower()
        x = self._tokenize(df.question.tolist())
        x = self._preprocess(x, self.seq_length)
        y = df.ix[:, 2:].as_matrix()
        y = self._preprocess(y, 5)

        for i in range(math.ceil(len(x)/batch_size)):
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

    def get_embedding(self):
        vocabulary = self.vocabulary
        
        w2v = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

        embedding = []
        sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1])

        vec = np.zeros((len(sorted_vocab), 300))
        for key, index in sorted_vocab:
            try:
                vec[index] = w2v[key]
            except KeyError:
                vec[index] = w2v['other']
        return vec.astype(np.float32)