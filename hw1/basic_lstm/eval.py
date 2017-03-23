import tensorflow as tf
import numpy as np
import argparse
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument('model_dir')

parser.add_argument('test_file')

parser.add_argument('output_file')

option = parser.parse_args()

model_dir = option.model_dir

test_file = option.test_file

output_file = option.output_file

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
with tf.Session(config=config) as sess:
    path = tf.train.latest_checkpoint(model_dir)
    
    saver = tf.train.import_meta_graph(path + '.meta')

    saver.restore(sess, path)
    # var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    data_loader = pickle.load(open(os.path.join(model_dir, 'data_loader.p'), 'rb'))
    
    choosed = []
    for x, y, p in data_loader.test_batch(40, test_file):
        feed = {'Placeholder:0': x, 'Placeholder_1:0': p}
        pred_out = sess.run('predict:0', feed_dict = feed)
        for p, index in zip(pred_out, y):
            choosed.append(np.argmax(p[index]))
    ans_list = ['a', 'b', 'c', 'd', 'e']

    with open(output_file, 'w') as out:
        out.write('id,answer')
        out.write('\n')
        for i, c in enumerate(choosed):
            out.write('%d,%s' % (i+1, ans_list[c]))
            out.write('\n')
        