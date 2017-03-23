import pickle
import collections
import numpy as np
import math
import gensim
from nltk.tokenize import RegexpTokenizer

class Reader():
    def __init__(self, voca_size, seq_length, min_length=5):
        self.voca_size = voca_size
        self.seq_length = seq_length
        self.min_length = min_length
        
        self.tokenizer = RegexpTokenizer(r'[\w#]+|[\w#]+\'[\w#]+')
        
        data = open('data/corpus2.txt', 'r').read().lower().split('\n')
        data = self._tokenize(data)
        
        split_num = int(len(data)/10)
        
        self.data = data
        self.training_data = data[:-split_num]
        self.validate_data = data[-split_num:]
        
        self.vocabulary, self.vocabulary_inv = self._build_voca()
        
    def _tokenize(self, data):
        tokenizer = self.tokenizer
        min_length = self.min_length
        
        data = [tokenizer.tokenize(s) for s in data]
        # seq_length-1 for start tag
        data = [s for s in data if len(s) >= min_length and len(s) <= self.seq_length-1]
        
        return data
    
    def _preprocess(self, data, is_test=False):
        vocabulary = self.vocabulary
        seq_length = self.seq_length
        
        data = [['<start>'] + s for s in data]
        
        data_id = np.zeros((len(data), seq_length)).astype(int)
        data_len = np.zeros(len(data)).astype(int)
        
        
        if is_test:
            for i, sent in enumerate(data):
                for j, word in enumerate(sent):
                    if word == '_____':
                        data_len[i] = j
                    data_id[i][j] = self._map_vocabulary(word)
                        
        else:
            for i, sent in enumerate(data):
                for j, word in enumerate(sent):
                    data_id[i][j] = self._map_vocabulary(word)
                    data_len[i] = len(sent)
                    
        return data_id, data_len
    
    def _map_vocabulary(self, word):
        vocabulary = self.vocabulary
        if word in vocabulary:
            return vocabulary[word]
        else:
            return vocabulary['<unknown>']
    
    def _build_voca(self):
        num_spare_words = 3
        data = self.data
        words = []
        for sent in data:
            words += sent
        counter = collections.Counter(words)

        vocabulary_inv = [x[0] for x in counter.most_common(self.voca_size-num_spare_words)]
        vocabulary_inv = list(sorted(vocabulary_inv))
        
        vocabulary = {x: i+num_spare_words for i, x in enumerate(vocabulary_inv)}
        vocabulary['<null>'] = 0
        vocabulary['<unknown>'] = 1
        vocabulary['<start>'] = 2
        vocabulary['<end>'] = 3
        return vocabulary, vocabulary_inv
    
    def train_batch(self, batch_size, shuffle=False):
        data = self.training_data
        vocabulary = self.vocabulary
        data_len = len(data)
        
        x, len_list = self._preprocess(data)
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        for i, l in enumerate(len_list):
            y[i][l-1] = 3
        
        num_batch = math.ceil(data_len/batch_size)
        order = np.arange(num_batch)
        if shuffle:
            np.random.shuffle(order)
            
        for i in order:
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            batch_l = len_list[i*batch_size:(i+1)*batch_size]
            yield batch_x, batch_y, batch_l
    
    def valid_batch(self, batch_size):
        data = self.validate_data
        vocabulary = self.vocabulary
        data_len = len(data)
        
        x, len_list = self._preprocess(data)
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        for i, l in enumerate(len_list):
            y[i][l-1] = 3
        
        num_batch = math.ceil(data_len/batch_size)
            
        for i in range(num_batch):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            batch_l = len_list[i*batch_size:(i+1)*batch_size]
            yield batch_x, batch_y, batch_l

    def test_batch(self, batch_size, file=None):
        import pandas as pd
        if not file:
            df = pd.read_csv('hw1_data/testing_data.csv')
        else:
            df = pd.read_csv(file)
        df.question = df.question.str.lower()
        
        
        x = self._tokenize(df.question.tolist())
        
        
        x, len_list = self._preprocess(x, is_test=True)
        y = df.ix[:, 2:].as_matrix()
        y = [[self._map_vocabulary(j) for j in i] for i in y]
        y = np.array(y).astype(int)

        for i in range(math.ceil(len(x)/batch_size)):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            batch_l = len_list[i*batch_size:(i+1)*batch_size]
            yield batch_x, batch_y, batch_l

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
                vec[index] = np.random.uniform(-1, 1, 300)
        vec[0] = np.random.uniform(-1, 1, 300)
        vec[1] = np.random.uniform(-1, 1, 300)
        vec[2] = np.random.uniform(-1, 1, 300)
        vec[3] = np.random.uniform(-1, 1, 300)
        return vec.astype(np.float32)
