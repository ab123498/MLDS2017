import numpy as np
import os
import pickle
import json
from nltk.tokenize import RegexpTokenizer
from collections import Counter

class Loader():
    def __init__(self, vocabulary_size=5000, max_length=40):
        path = 'data/vocabulary%d.p' % vocabulary_size
        self.vocabulary_size = vocabulary_size
        self.voca = self.build_voca(path)
        self.max_length= max_length
        
    def build_voca(self, path):
        _size = self.vocabulary_size
        if not os.path.exists(path):
            with open('./MLDS_hw2_data/training_label.json') as json_data:
                training_label = json.load(json_data)

            sents = []
            for o in training_label:
                sents += o['caption']

            words = []
            tokenizer = RegexpTokenizer(r'[\w\'\-\d]+')
            for sent in sents:
                words += tokenizer.tokenize(sent)

            words = [w.lower() for w in words]

            counter = Counter(words)
        
            vocabulary = {x[0]: i+4 for i, x in enumerate(counter.most_common(_size-4))}
            vocabulary['<pad>'] = 0
            vocabulary['<bos>'] = 1
            vocabulary['<eos>'] = 2
            vocabulary['<unknown>'] = 3
            
            pickle.dump(vocabulary, open(path, 'wb'))
        else:
            vocabulary = pickle.load(open(path, 'rb'))
        
        return vocabulary


    def train_data(self, batch_size, shuffle=True):
        with open('./MLDS_hw2_data/training_label.json') as f:
            training_label = json.load(f)

        caption = []
        feat_file = []
        tokenizer = RegexpTokenizer(r'[\w\'\-\d]+')
        for pair in training_label:
            caption += [tokenizer.tokenize(s) for s in pair['caption']]
            feat_file += [pair['id']] * len(pair['caption'])
        feat_file = np.array(feat_file)
        
        raw_caption = []
        for c in caption:
            row = []
            for i in c:
                try:
                    v = self.voca[i.lower()]
                except KeyError:
                    v = self.voca['<unknown>']
                row.append(v)
            row.append(2)
            raw_caption.append(row)
        
        caption = np.zeros([len(caption), self.max_length], int)
        for i, cap in enumerate(raw_caption):
            cap = cap[:self.max_length]
            caption[i, :len(cap)] = cap
        
        if shuffle:
            order = np.arange(len(caption))
            np.random.shuffle(order)
            feat_file = feat_file[order]
            caption = caption[order]
        
        batch_num = int(len(caption)/batch_size)
        for i in range(batch_num):
            batch_feat_file = feat_file[i*batch_size:(i+1)*batch_size]
            feat_dir = './MLDS_hw2_data/training_data/feat/%s.npy'

            batch_feat = np.zeros([len(batch_feat_file), 80, 4096])
            for index, f in enumerate(batch_feat_file):
                batch_feat[index] = np.load(feat_dir % f)
            batch_caption = caption[i*batch_size:(i+1)*batch_size]

            yield batch_feat, batch_caption