import re
import os
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
tokenizer = PunktSentenceTokenizer(punkt_param)

def remove_header(file):
    for line in file:
        if '*END*THE SMALL PRINT!' in line:
            end = True
            break
    text = file.read()
    file.close()
    return text

def filter_paragraph(text):
    text = re.split('\n\n', text)
    text = [re.sub('\n', ' ', s).strip() for s in text]
    text = [s for s in text if re.search(r'\.$', s)]
    return text

def parse_sent(text):
    ret = []
    for line in text:
        l = re.findall(r'\"[^"]+\"', line)
        line = re.sub(r'\"[^"]+\"', 'SOME_THING', line)
        
        tokens = tokenizer.tokenize(line)
        new_tokens = []
        for token in tokens:
#             while re.search(r'SOME_THING', token):
#                 token = re.sub(r'SOME_THING', l[0], token, count=1)
#                 del l[0]
            if re.search(r'SOME_THING', token):
                continue
            new_tokens.append(token)
        new_tokens += l
        ret += new_tokens
    return ret

dir_path = 'hw1_data/Holmes_Training_Data/'
output_file = open('data/corpus2.txt', 'w')

for f in os.listdir(dir_path):
    print(f)
    file = open(dir_path + f, 'r', errors='ignore')
    text = remove_header(file)
    text = filter_paragraph(text)
    text = parse_sent(text)
    output = open('data/corpus2/' + f, 'w')
    for sent in text:
        output_file.write(sent + '\n')
output_file.close()