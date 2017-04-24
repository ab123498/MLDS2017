import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from loader import Loader
import time
import os
import pickle

# hyper parameter
video_length = 80

image_dim = 4096
n_voca = 3000

dim_hidden = 500

max_caption_length = 20
batch_size = 32
learning_rate = 0.001
n_epoch = 50


video = tf.placeholder(tf.float32, [None, video_length, image_dim])
caption = tf.placeholder(tf.int32, [None, max_caption_length])

_batch_size = tf.shape(video)[0]


encode_img_w = tf.Variable(tf.random_uniform([image_dim, dim_hidden], -0.1, 0.1))
encode_img_b = tf.Variable(tf.zeros([dim_hidden]))
v_flat = tf.reshape(video, [-1, image_dim])
image_feat = tf.matmul(v_flat, encode_img_w) + encode_img_b
image_feat = tf.reshape(image_feat, [_batch_size, video_length, dim_hidden])

word_embedding = tf.Variable(tf.random_uniform([n_voca, dim_hidden], -0.1, 0.1))
embed_word_w = tf.Variable(tf.random_uniform([dim_hidden, n_voca], -0.1, 0.1))
embed_word_b = tf.Variable(tf.zeros([n_voca]))

lstm = rnn.BasicLSTMCell(dim_hidden)

""" encoding """
with tf.variable_scope('LSTM1'):    
    output1, state1 = tf.nn.dynamic_rnn(lstm, image_feat, dtype=tf.float32)
    
with tf.variable_scope('LSTM2'):
    padding = tf.zeros([_batch_size, video_length, dim_hidden], dtype=tf.float32)
    rnn_input = tf.concat([padding, output1], 2)

    output2, state2 = tf.nn.dynamic_rnn(lstm, rnn_input, dtype=tf.float32)

""" decoding """
with tf.variable_scope('LSTM1', reuse=True):
    padding = tf.zeros([_batch_size, max_caption_length, dim_hidden])
    output1, state1 = tf.nn.dynamic_rnn(lstm, padding, dtype=tf.float32)

with tf.variable_scope('LSTM2'):
    # for bos(1)
    word = tf.ones([_batch_size], tf.int32)
    output1 = tf.split(output1, max_caption_length, 1)
    predict = []
    for i in range(max_caption_length):
        with tf.device("/cpu:0"):
            w_emb = tf.nn.embedding_lookup(word_embedding, word)

        _lstm_output = tf.reshape(output1[i], [_batch_size, dim_hidden])
        rnn_input = tf.concat([w_emb, _lstm_output], 1)
        with tf.variable_scope('rnn', reuse=True):
            output2, state2 = lstm(rnn_input, state2)
        
        logit = tf.matmul(output2, embed_word_w) + embed_word_b
        predict.append(logit)

        word = tf.arg_max(logit, 1)

target = tf.transpose(caption, [1, 0])
target = tf.split(target, max_caption_length, 0)

seq_cost = 0
for p, t in zip(predict, target):
    t = tf.reshape(t, [-1])
    
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p, labels=t)
    cost = tf.reduce_mean(cost)
    
    seq_cost += cost

seq_cost = seq_cost / max_caption_length

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(seq_cost)


predict = [tf.reshape(p, [_batch_size, 1, n_voca]) for p in predict]
seq_output = tf.concat(predict, 1)
seq_output = tf.arg_max(seq_output, 2, name='output')

loader = Loader(vocabulary_size=n_voca, max_length=max_caption_length)

output_path = './basic_model'
if not os.path.exists(output_path):
    os.mkdir(output_path)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    tf.add_to_collection('output', seq_output)
    
    saver = tf.train.Saver(max_to_keep=100)
    
    pickle.dump(loader, open('./basic_model/loader.p', 'wb'))
    
    for epoch in range(n_epoch):
        print('Epoch :', epoch)
        start = time.time()
        total_cost = 0
        for i, (feat, cap) in enumerate(loader.train_data(batch_size)):
            feed = {
                video: feat,
                caption: cap
            }
            _, cost = sess.run([optimizer, seq_cost], feed_dict=feed)
            total_cost += cost
            if (i+1) % 100 == 0:
                print('Time :', time.time() - start, 'Loss :', total_cost/(i+1))
        
        avg_cost = total_cost/(i+1)
        print('Time :', time.time() - start, 'Loss :', avg_cost)
        
        saver.save(sess, './basic_model/epoch%d-%.2f' % (epoch+1, avg_cost))
