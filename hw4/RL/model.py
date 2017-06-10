import tensorflow as tf
import numpy as np

class Seq2Seq():
    def __init__(self,
                 max_seq_len,
                 voca_size,
                 embed_size,
                 rnn_size,
                 n_layers):
        self.max_seq_len = max_seq_len
        self.voca_size = voca_size
        self.embed_size = embed_size
        self.rnn_size = rnn_size
        self.n_layers = n_layers

    def build_model(self, train=True):
        max_seq_len = self.max_seq_len
        voca_size = self.voca_size
        embed_size = self.embed_size
        rnn_size = self.rnn_size
        n_layers = self.n_layers

        # Input Tensors
        text_input = tf.placeholder(tf.int32, [None, max_seq_len])

        input_len = tf.placeholder(tf.int32, [None])

        target = tf.placeholder(tf.int32, [None, max_seq_len])

        reward = tf.placeholder(tf.float32, [None])

        batch_size = tf.shape(text_input)[0]

        with tf.variable_scope('embed'):
            embedding = tf.Variable(tf.random_uniform([voca_size, embed_size]))
            

        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm] * n_layers)

        # encoder
        encoder_input = tf.unstack(text_input, max_seq_len, 1)
        encoder_input = [tf.nn.embedding_lookup(embedding, i) for i in encoder_input]
        
        _, state = tf.contrib.rnn.static_rnn(rnn_cell, encoder_input, sequence_length=input_len, dtype=tf.float32)

        with tf.variable_scope('project'):
            proj_w = tf.Variable(tf.random_uniform([rnn_size, voca_size]))
            proj_b = tf.Variable(tf.zeros([voca_size]))

        def projectOp(output):
            return tf.matmul(output, proj_w) + proj_b

        # decoder
        unstack_target = tf.unstack(target, max_seq_len, 1)
        decoder_output = []
        with tf.variable_scope('rnn', reuse=True):
            inp = tf.nn.embedding_lookup(embedding, tf.tile([1], [batch_size]))
            output, state = rnn_cell(inp, state)
            output = projectOp(output)
            decoder_output.append(output)

            # if train:
            #     for inp in unstack_target:
            #         inp = tf.nn.embedding_lookup(embedding, inp)
            #         output, state = rnn_cell(inp, state)
            #         output = projectOp(output)
            #         decoder_output.append(output)
            # else:
            #     for _ in range(max_seq_len):
            #         inp = tf.argmax(output, 1)
            #         inp = tf.nn.embedding_lookup(embedding, inp)
            #         output, state = rnn_cell(inp, state)
            #         output = projectOp(output)
            #         decoder_output.append(output)
            for _ in range(max_seq_len):
                inp = tf.reshape(tf.multinomial(output, 1), [-1])
                inp = tf.nn.embedding_lookup(embedding, inp)
                output, state = rnn_cell(inp, state)
                output = projectOp(output)
                decoder_output.append(output)

        loss = 0
        text_output = []
        for pred, targ in zip(decoder_output, unstack_target):
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pred,
                labels=targ
            )
            text_output.append(tf.argmax(pred, 1))

        loss *= reward
        loss = tf.reduce_mean(loss/max_seq_len)
        text_output = tf.stack(text_output, axis=1)

        input_tensors = {
            'text_input': text_input,
            'input_len': input_len,
            'target': target,
            'reward': reward
        }

        output_tensors = {
            'loss': loss,
            'text_output': text_output
        }

        return input_tensors, output_tensors

if __name__ == '__main__':
    s2s = Seq2Seq()
    s2s.build_model()
    tf.reset_default_graph()
    s2s.build_model(False)



