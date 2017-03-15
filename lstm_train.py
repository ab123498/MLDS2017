import tensorflow as tf
import numpy as np
from utils import DataLoader
from tensorflow.contrib import rnn


seq_length = 34
batch_size = 128
rnn_size = 256
voca_size = 40000
learning_rate = 1e-2
num_layers = 3
num_epochs = 100

data_loader = DataLoader(voca_size, seq_length)

input_data = tf.placeholder(tf.int32, [None, seq_length])
target = tf.placeholder(tf.int32, [None])

with tf.device('/cpu:0'):
    embedding_vec = tf.constant(data_loader.get_embedding())
#     embedding_vec = tf.random_uniform([voca_size, 300], -1.0, 1.0)
    embedded_input = tf.nn.embedding_lookup(embedding_vec, input_data)


def RNN(x):
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, 300])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, seq_length, 0)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)
    
    cell = rnn.MultiRNNCell([lstm_cell] * num_layers)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return outputs[-1]

output = RNN(embedded_input)

w = tf.Variable(tf.random_normal([rnn_size, voca_size]))
b = tf.Variable(tf.random_normal([voca_size]))
pred = tf.matmul(output, w) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=target))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Test model
# correct_prediction = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), target)
# # Calculate accuracy
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (x, y) in enumerate(data_loader.train_batch(batch_size)):
            feed = {input_data: x, target: y}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            total_loss += loss
        print('Epoch', epoch, 'loss:', total_loss/(i+1))
        
        if (epoch+1) % 10 == 0:
            choosed = []
            for x, y in data_loader.test_batch(10):
                pred_out = pred.eval({input_data: x})
                for p, index in zip(pred_out, y):
                    choosed.append(np.argmax(p[index]))
            ans_list = ['a', 'b', 'c', 'd', 'e']
            with open('submit/test-%d.csv' % epoch, 'w') as out:
                out.write('id,answer')
                out.write('\n')
                for i, c in enumerate(choosed):
                    out.write('%d,%s' % (i+1, ans_list[c]))
                    out.write('\n')
    

