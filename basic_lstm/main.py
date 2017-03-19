import tensorflow as tf
import numpy as np
from reader import Reader
from tensorflow.contrib import rnn, legacy_seq2seq
import time
from datetime import datetime
import os
import pickle

seq_length = 40
batch_size = 64
rnn_size = 256
voca_size = 25000
learning_rate = 1e-2
num_layers = 3
num_epochs = 20

print("prepare data......")
data_loader = Reader(voca_size, seq_length)

input_data = tf.placeholder(tf.int32, [None, seq_length])
step = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None, seq_length])

with tf.device('/cpu:0'):
    print("prepare embedding......")
    embedding_vec = tf.constant(data_loader.get_embedding(), dtype=tf.float32)
    # embedding_vec = tf.random_uniform([voca_size, 300], -1.0, 1.0)
    embedded_input = tf.nn.embedding_lookup(embedding_vec, input_data)



# Define a lstm cell with tensorflow
lstm_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)

cell = rnn.MultiRNNCell([lstm_cell] * num_layers)

# Get lstm cell output
outputs, states = tf.nn.dynamic_rnn(cell, embedded_input, dtype=tf.float32, sequence_length=step)

last_output = states[num_layers-1].h


t_outputs = tf.transpose(outputs, [1, 0, 2])
t_outputs = tf.split(t_outputs, seq_length, 0)
t_target = tf.transpose(target, [1, 0])
t_target = tf.split(t_target, seq_length, 0)

total_cost = 0
l_w = tf.Variable(tf.random_normal([rnn_size, voca_size]))
l_b = tf.Variable(tf.zeros([voca_size]) + 0.1)
for p, t in zip(t_outputs, t_target):
    p = tf.reshape(p, [-1, rnn_size])
    t = tf.reshape(t, [-1])    
    logit = tf.matmul(p, l_w) + l_b

    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=t)

    cost = tf.reduce_mean(cost)

    total_cost += cost

total_cost = total_cost / seq_length

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

pred = tf.matmul(last_output, l_w) + l_b

init = tf.global_variables_initializer()

t = datetime.now().strftime('%Y%m%d_%H%M/')
output_dir = 'basic_lstm/' + t

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    for epoch in range(num_epochs):
        train_loss = 0
        start_time = time.time()
        print("training......")
        for i, (x, y, p) in enumerate(data_loader.train_batch(batch_size, shuffle=True)):
            feed = {input_data: x, target: y, step: p}
            loss, _ = sess.run([total_cost, optimizer], feed_dict=feed)
            train_loss += loss
            if (i+1) % 100 == 0:
                print('Iter:', i+1, 'loss:', train_loss/(i+1))
            if i >= 5000:
                break
        avg_train_loss = train_loss/(i+1)
        
        valid_loss = 0
        print('validating......')
        for i, (x, y, p) in enumerate(data_loader.valid_batch(batch_size)):
            feed = {input_data: x, target: y, step: p}
            loss = sess.run(total_cost, feed_dict=feed)
            valid_loss += loss
            if (i+1) % 100 == 0:
                print(valid_loss/(i+1))
            if i >= 500:
                break
        avg_valid_loss = valid_loss/(i+1)
        print("--- %s seconds ---" % (time.time() - start_time))        
        print('Epoch', epoch, 'train loss:', avg_train_loss, 'valid loss:', avg_valid_loss)
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            pickle.dump(data_loader.vocabulary, open(output_dir + 'vocabulary.p', 'wb'))
        saver.save(sess, output_dir + 'epoch%d-%.2f' % (epoch, avg_valid_loss))
