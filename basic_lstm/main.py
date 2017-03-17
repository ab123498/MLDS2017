import tensorflow as tf
import numpy as np
from reader import Reader
from tensorflow.contrib import rnn
import time

seq_length = 40
batch_size = 128
rnn_size = 256
voca_size = 40000
learning_rate = 1e-2
num_layers = 3
num_epochs = 20

print("prepare data......")
data_loader = Reader(voca_size, seq_length)

input_data = tf.placeholder(tf.int32, [None, seq_length])
input_len = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None])

with tf.device('/cpu:0'):
    print("prepare embedding......")
    # embedding_vec = tf.Variable(data_loader.get_embedding(), dtype=tf.float32)
    embedding_vec = tf.random_uniform([voca_size, 300], -1.0, 1.0)
    embedded_input = tf.nn.embedding_lookup(embedding_vec, input_data)


def RNN(x):
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)
    
    cell = rnn.MultiRNNCell([lstm_cell] * num_layers)

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=input_len-1)
    
    output = states[num_layers-1].h
    
    return output

output = RNN(embedded_input)

nce_w = tf.Variable(tf.random_normal([voca_size, rnn_size]))
nce_b = tf.Variable(tf.zeros([voca_size]) + 0.1)

nce_w_t = tf.transpose(nce_w, [1, 0])

pred = tf.matmul(output, nce_w_t) + nce_b

label = tf.reshape(target, [-1, 1])

nce = tf.nn.sampled_softmax_loss(weights=nce_w, biases=nce_b,
                     labels=label, inputs=output,
                     num_sampled=100, num_classes=voca_size)
cost = tf.reduce_mean(nce)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    best_loss = 999
    print("training......")
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        for i, (x, y, p) in enumerate(data_loader.train_batch(batch_size, shuffle=True)):
            feed = {input_data: x, target: y, input_len: p}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            total_loss += loss
            if (i+1) % 10 == 0:
                print(total_loss/(i+1))
        print("--- %s seconds ---" % (time.time() - start_time))
        avg_loss = total_loss/(i+1)
        print('Epoch', epoch, 'loss:', avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            choosed = []
            for x, y, p in data_loader.test_batch(40):
                print(x.shape)
                pred_out = pred.eval({input_data: x, input_len: p})
                for p, index in zip(pred_out, y):
                    choosed.append(np.argmax(p[index]))
            ans_list = ['a', 'b', 'c', 'd', 'e']
            with open('submit/epoch%d-%.2f.csv' % (epoch, avg_loss), 'w') as out:
                out.write('id,answer')
                out.write('\n')
                for i, c in enumerate(choosed):
                    out.write('%d,%s' % (i+1, ans_list[c]))
                    out.write('\n')
