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
        
        training_data = open('data/corpus2.txt', 'r').read().lower().split('\n')
        self.training_data = self._tokenize(training_data)
        
        self.vocabulary, self.vocabulary_inv = self._build_voca()
        
    def _tokenize(self, data):
        tokenizer = self.tokenizer
        min_length = self.min_length
        
        data = [tokenizer.tokenize(s) for s in data]
        # seq_length-1 for start tag
        data = [s for s in data if len(s) >= min_length and len(s) <= self.seq_length-1]
        
        return data
    
    def _preprocess(self, data):
        vocabulary = self.vocabulary
        seq_length = self.seq_length
        
        data = [['<start>'] + s for s in data]
        
        data_id = np.zeros((len(data), seq_length)).astype(int)
        len_list = np.zeros(len(data)).astype(int)
        for i, sent in enumerate(data):
            for j, word in enumerate(sent):
                data_id[i][j] = self._map_vocabulary(word)
                # get place to fill
                if word == '_____':
                    len_list[i] = j
            # whole sequence length
            # if len(sent) < seq_length:
            #     len_list[i] = len(sent)
            # else:
            #     len_list[i] = seq_length
        return data_id, len_list
    
    def _map_vocabulary(self, word):
        vocabulary = self.vocabulary
        if word in vocabulary:
            return vocabulary[word]
        else:
            return vocabulary['<unknown>']
    
    def _build_voca(self):
        num_spare_words = 4
        data = self.training_data
        words = []
        for sent in data:
            words += sent
        counter = collections.Counter(words)

        vocabulary_inv = [x[0] for x in counter.most_common(self.voca_size-num_spare_words)]
        vocabulary_inv = list(sorted(vocabulary_inv))
        
        vocabulary = {x: i+num_spare_words for i, x in enumerate(vocabulary_inv)}
        vocabulary['<null>'] = 0
        vocabulary['_____'] = 1
        vocabulary['<unknown>'] = 2
        vocabulary['<start>'] = 3
        return vocabulary, vocabulary_inv
    
    def train_batch(self, batch_size, shuffle=False):
        data = self.training_data
        vocabulary = self.vocabulary
        data_len = len(data)
        
        target_word = []
        
        for i, sent in enumerate(data):
            l = len(sent)
            place2fill = np.random.randint(math.floor(l/4), math.ceil(l/4*3))

            target_word.append(sent[place2fill])
            data[i][place2fill] = '_____'
        
        x, len_list = self._preprocess(data)
        y = [self._map_vocabulary(i) for i in target_word]
        y = np.array(y).astype(int)
        
        num_batch = math.ceil(data_len/batch_size)
        order = np.arange(num_batch)
        if shuffle:
            np.random.shuffle(order)
            
        
        for i in order:
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            batch_l = len_list[i*batch_size:(i+1)*batch_size]
            yield batch_x, batch_y, batch_l

    def test_batch(self, batch_size):
        import pandas as pd
        df = pd.read_csv('hw1_data/testing_data.csv')
        df.question = df.question.str.lower()
        
        
        x = self._tokenize(df.question.tolist())
        
        
        x, len_list = self._preprocess(x)
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
                vec[index] = np.zeros(300)
        vec[1] = np.zeros(300)
        return vec.astype(np.float32)
