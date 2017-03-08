import os
from nltk.tokenize import RegexpTokenizer
import re
import pickle

sents = []
for filename in os.listdir('hw1_data/Holmes_Training_Data/'):
    with open('hw1_data/Holmes_Training_Data/' + filename, 'r', errors='ignore') as f:
        text = f.read()

        oral = re.findall(r'\"[\w\s,\'.]+[.,?!]\"', text)

        oral = [re.sub(r'\s+', ' ', t) for t in oral]

        oral = [t for t in oral if not re.search("Small Print!", t, re.IGNORECASE)]

        sents += oral
        
#        with open('data/corpus/' + filename, 'w') as out:
#            for o in oral:
#                out.write(o + '\n')

tokenizer = RegexpTokenizer(r'[\w\']+')

sents = [[word.lower() for word in tokenizer.tokenize(s)] for s in sents]

pickle.dump(sents, open('data/tokenized_corpus.p', 'wb'))
