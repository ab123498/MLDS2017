from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np


class Loader:
    def __init__(self, voca_size=20000, max_seq_len=20):
        self.voca_size = voca_size
        self.max_seq_len = max_seq_len

        self.questions = self.parse_raw(open('processed_data/cornell_question.txt').readlines())
        self.answers = self.parse_raw(open('processed_data/cornell_answer.txt').readlines())

        self.dic = self.build_dictionary()

    def parse_raw(self, lines):
        tokenizer = RegexpTokenizer(
            r'[\w\']+|\?|[\.]+|,|\!'
        )

        lines = [tokenizer.tokenize(_.lower()) for _ in lines]
        return lines


    def build_dictionary(self):
        voca_size = self.voca_size
        words = [w for l in (self.questions+self.answers) for w in l]

        counter = Counter(words)

        dic = {c[0]: i+4 for i, c in enumerate(counter.most_common(voca_size-4))}

        # special symbol
        dic['<pad>'] = 0
        dic['<bos>'] = 1
        dic['<eos>'] = 2
        dic['<unk>'] = 3

        return dic


    def train_data(self, batch_size=30, shuffle=True):
        questions = self.questions
        answers = self.answers
        to_id = self.to_id


        n_conv = len(questions)
        
        ques_ids, ques_len = to_id(questions)
        ans_ids, ans_len = to_id(answers)

        order = np.random.shuffle(np.arange(n_conv))
        n_batch = int((n_conv-1)/batch_size + 1)

        for i in range(n_batch):
            yield (ques_ids[batch_size*i:batch_size*(i+1)],
                   ques_len[batch_size*i:batch_size*(i+1)],
                   ans_ids [batch_size*i:batch_size*(i+1)])

    def to_id(self, lines):
        dic = self.dic
        max_seq_len = self.max_seq_len

        n_conv = len(lines)

        ids = np.zeros([n_conv, max_seq_len], dtype=int)

        # <unk> = 3
        id_lines = [[dic[w] if w in dic else 3 for w in l] for l in lines] 

        # add <eos>
        id_lines = [l + [2] for l in id_lines] 

        lens = []
        for i in range(n_conv):
            l = id_lines[i][:max_seq_len]
            l_len = len(l)
            ids[i, :l_len] = l
            lens.append(l_len)
        lens = np.array(lens)
        return ids, lens

    def to_test(self, lines):
        lines = self.parse_raw(lines)
        return self.to_id(lines)


if __name__ == '__main__':
    l = Loader()
    next(l.train_data())
    test = ['how are you', 'test test', 'nice to see you']
    print(l.to_test(test))
