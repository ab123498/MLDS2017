import tensorflow as tf
import numpy as np
import pickle
import os
import string
from id2sent import id2sent
import sys

def main():
    model_path = 'RL_model'
    if not os.path.exists(model_path):
        print('trained model not exists!!')

    s2s = pickle.load(open(os.path.join(model_path, 's2s.p'), 'rb'))
    loader = pickle.load(open(os.path.join(model_path, 'loader.p'), 'rb'))

    dic = loader.dic

    input_tensors, output_tensors = s2s.build_model(train=False)

    test_data = open(sys.argv[1]).readlines()
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

        outputs = id2sent(dic, output)
        
        with open(sys.argv[2], 'w') as f:
            for o in outputs:
                f.write(o + '\n')
        


if __name__ == '__main__':
    main()




