import tensorflow as tf
import numpy as np
import pickle
import os
import string

model_path = 'seq2seq/models'
if not os.path.exists(model_path):
    print('trained model not exists!!')

s2s = pickle.load(open(os.path.join(model_path, 's2s.p'), 'rb'))
loader = pickle.load(open(os.path.join(model_path, 'loader.p'), 'rb'))

dic = loader.dic
inv_dic = {d[1]:d[0] for d in dic.items()}

input_tensors, output_tensors = s2s.build_model(train=False)

test_data = open('sample_input.txt').readlines()
test_id, test_len = loader.to_test(test_data)

config = tf.ConfigProto(
            device_count = {'GPU': 0}
         )
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    saver_path = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, saver_path)

    text_input = input_tensors['text_input']
    input_len = input_tensors['input_len']

    text_output = output_tensors['text_output']
    feed = {text_input: test_id, input_len: test_len}
    output = sess.run(text_output, feed_dict=feed)

    for l in output:
        l = [inv_dic[i] for i in l]

        cap = False
        s = string.capwords(l[0])
        for w in l[1:]:
            if w == '<eos>':
                break
            elif w == '<pad>':
                continue
            elif w == '.':
                cap = True
                s += w
            elif w in set(string.punctuation):
                s += w
            else:
                if cap:
                    w = string.capwords(w)
                    cap = False
                s += ' ' + w

        print(s)







