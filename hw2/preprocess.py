import numpy as np
import json
import os
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from collections import Counter, OrderedDict
import numpy as np


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

vocabulary = {x[0]: i+4 for i, x in enumerate(counter.most_common())}
vocabulary['<pad>'] = 0
vocabulary['<bos>'] = 1
vocabulary['<eos>'] = 2
vocabulary['<unknown>'] = 3

print(len(vocabulary))
pickle.dump(vocabulary, open('data/vocabulary.p', 'wb'))
