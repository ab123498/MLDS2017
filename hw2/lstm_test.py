import tensorflow as tf
import numpy as np
import pickle
import os
import argparse
import math
import json

# parser = argparse.ArgumentParser()
# parser.add_argument('testing_list')
# parser.add_argument('feat_path')
# option = parser.parse_args()


def test_data(batch_size):
    testing_dir = './MLDS_hw2_data/testing_data/feat'
    testing_list = './MLDS_hw2_data/testing_id.txt'
    testing_id = open(testing_list).read().strip().split('\n')
    feat_file = [(i, os.path.join(testing_dir, "%s.npy" % i)) for i in testing_id]

    batch_num = int(math.floor(len(feat_file)/batch_size))

    for i in range(batch_num):
        batch_feat_file = feat_file[i*batch_size:(i+1)*batch_size]

        batch_feat = np.zeros([len(batch_feat_file), 80, 4096])
        batch_id = []
        for index, (i, f) in enumerate(batch_feat_file):
            batch_id.append(i)
            batch_feat[index] = np.load(f)

        yield batch_id, batch_feat

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
with tf.Session(config=config) as sess:
    model_dir = './basic_model'
    path = tf.train.latest_checkpoint(model_dir)    
    saver = tf.train.import_meta_graph(path + '.meta')

    saver.restore(sess, path)
    
    loader = pickle.load(open(os.path.join(model_dir, 'loader.p'), 'rb'))
    
    voca = loader.voca
    inv_voca = {v: k for k, v in voca.items()}
    
    output = []
    for id, feat in test_data(10):
        feed = {'Placeholder:0': feat}
        
        output_op = tf.get_collection('output')[0]
        
        pred = output_op.eval(feed)
    
        for i, row in zip(id, pred):
            seq = ''
            for word in row:
                word = inv_voca[word]
                if word == '<eos>':
                    break
                if word == '<pad>':
                    word = ''
                if word == '<unknown>':
                    word = ''
                seq += ' ' + word
            seq = seq.strip()
            d = {'caption': seq, 'id': i}
            output.append(d)
            
    open('output.json', 'w').write(json.dumps(output))
    
            
    