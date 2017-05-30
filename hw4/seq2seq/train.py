import tensorflow as tf
import numpy as np
from model import Seq2Seq
from loader import Loader
import pickle
import os
import shutil


def main(restore=False):
    max_seq_len = 20
    voca_size = 20000
    embed_size = 300
    rnn_size = 256
    n_layers = 3

    n_epoch = 30
    batch_size = 32
    learning_rate = 1e-3


    s2s = Seq2Seq(max_seq_len, voca_size, embed_size, rnn_size, n_layers)

    input_tensors, output_tensors = s2s.build_model()

    text_input = input_tensors['text_input']
    input_len = input_tensors['input_len']
    target = input_tensors['target']

    loss = output_tensors['loss']
    text_output = output_tensors['text_output']

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    voca_size = s2s.voca_size
    max_seq_len = s2s.max_seq_len

    if restore:
        loader = pickle.load(open('seq2seq/models/loader.p', 'rb'))
    else:
        if os.path.exists('models'):
            shutil.rmtree('models')
        os.mkdir('models')

        loader = Loader(voca_size, max_seq_len)
        # store the loader for test
        pickle.dump(loader, open('seq2seq/models/loader.p', 'wb'))# 

    # save graph model for test
    pickle.dump(s2s, open('seq2seq/models/s2s.p', 'wb'))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        saver = tf.train.Saver()
        if restore:
            saver_path = tf.train.latest_checkpoint('models')
            saver.restore(sess, saver_path)

        for epoch in range(n_epoch):
            batch_loss = []
            for i, (ques, lens, ans) in enumerate(loader.train_data(batch_size=batch_size)):
                
                feed = {
                    text_input: ques,
                    input_len: lens,
                    target: ans
                }

                _loss, _ = sess.run([loss, optimizer], feed_dict=feed)

                batch_loss.append(_loss)

                if (i+1) % 100 == 0:
                    print(np.mean(batch_loss))

            epoch_loss = np.mean(batch_loss)
            print('Epoch %s Loss: %s' % (epoch, epoch_loss))
            with open('loss_log', 'a') as f:
                f.write('%s\n' % epoch_loss)
            saver.save(sess, "seq2seq/models/model_epoch_%d.ckpt" % epoch)

if __name__ == '__main__':
    main()