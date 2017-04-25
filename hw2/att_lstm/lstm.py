import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from loader import Loader
import time
import os
import pickle
import math
import json

# hyper parameter
video_length = 80

image_dim = 4096
n_voca = 5000

dim_hidden = 500

max_caption_length = 20
batch_size = 50
learning_rate = 0.0001
n_epoch = 500

def model(test=False, true_caption=True, embedding=None):
    video = tf.placeholder(tf.float32, [None, video_length, image_dim])
    caption = tf.placeholder(tf.int64, [None, max_caption_length])
    caption_mask = tf.placeholder(tf.float32, [None, max_caption_length])
    
    _batch_size = tf.shape(video)[0]


    encode_img_w = tf.Variable(tf.random_uniform([image_dim, dim_hidden], -0.1, 0.1))
    encode_img_b = tf.Variable(tf.zeros([dim_hidden]))
    v_flat = tf.reshape(video, [-1, image_dim])
    image_feat = tf.matmul(v_flat, encode_img_w) + encode_img_b
    image_feat = tf.reshape(image_feat, [_batch_size, video_length, dim_hidden])
    
    if embedding is not None:
        word_embedding = tf.Variable(embedding, dtype=tf.float32)
    else:
        word_embedding = tf.Variable(tf.random_uniform([n_voca, dim_hidden], -0.1, 0.1))
    
    embed_word_w = tf.Variable(tf.random_uniform([dim_hidden, n_voca], -0.1, 0.1))
    embed_word_b = tf.Variable(tf.zeros([n_voca]))

    lstm = rnn.BasicLSTMCell(dim_hidden)

    """ encoding """
    with tf.variable_scope('LSTM1'):
        seq_len = tf.fill([_batch_size], video_length)
        output1, state1 = tf.nn.bidirectional_dynamic_rnn(lstm, lstm, image_feat, seq_len, dtype=tf.float32)
        output_fw1, output_bw1 = output1
        encoding_output = tf.concat([output_fw1, output_bw1], 2)
        
    """ attention """
    with tf.variable_scope('ATT'):
        att_list = []
        reshape_encoding = tf.reshape(encoding_output, [_batch_size, video_length*dim_hidden*2])
        for i in range(max_caption_length):
            att_w = tf.Variable(tf.random_uniform([video_length*dim_hidden*2, video_length]))
            att_b = tf.Variable(tf.zeros([video_length]))
            probs = tf.matmul(reshape_encoding, att_w) + att_b
            probs = tf.nn.softmax(probs)
            att_list.append(probs)


    """ decoding """
    
    random_sample = tf.placeholder(tf.bool, [max_caption_length])
    with tf.variable_scope('LSTM2') as scope:
        # for bos(1)
        word = tf.ones([_batch_size], tf.int32)
        predict = []
        
        state2 = lstm.zero_state(_batch_size, tf.float32)
        for i in range(max_caption_length):
            if i > 0:
                scope.reuse_variables()
            
            with tf.device("/cpu:0"):
                w_emb = tf.nn.embedding_lookup(word_embedding, word)
                
            att = tf.reshape(att_list[i], [_batch_size, video_length, 1])
            rnn_input = tf.reduce_sum(tf.multiply(encoding_output, att) , 1)
            rnn_input = tf.concat([w_emb, rnn_input], 1)

            with tf.variable_scope('rnn'):
                output2, state2 = lstm(rnn_input, state2)

            logit = tf.matmul(output2, embed_word_w) + embed_word_b
            predict.append(logit)
            if test or not true_caption:
                word = tf.arg_max(logit, 1)
            else:
                word = tf.cond(random_sample[i], lambda: caption[:, i], lambda: tf.arg_max(logit, 1))

    """output"""
    if test:
        predict = [tf.reshape(p, [_batch_size, 1, n_voca]) for p in predict]
        seq_output = tf.concat(predict, 1)
        seq_output = tf.arg_max(seq_output, 2, name='output')
        
        return video, seq_output
    
    target = tf.transpose(caption, [1, 0])
    target = tf.split(target, max_caption_length, 0)

    seq_cost = 0
    for i, (p, t) in enumerate(zip(predict, target)):
        t = tf.reshape(t, [-1])

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p, labels=t)
        cost = cost * caption_mask[:, i]
        
        cost = tf.reduce_sum(cost)
        seq_cost += cost

    seq_cost = seq_cost / tf.reduce_sum(caption_mask)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(seq_cost)

    # return video, caption, optimizer, seq_cost, random_sample
    return video, caption, caption_mask, optimizer, seq_cost, random_sample

def train():
    output_path = './att_model/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    loader = Loader(vocabulary_size=n_voca, max_length=max_caption_length)
    pickle.dump(loader, open(output_path + 'loader.p', 'wb'))
    
    # embedding = loader.get_embedding()
    # video, caption, optimizer, seq_cost, random_sample = model(embedding=embedding)
    
    # video, caption, optimizer, seq_cost, random_sample = model()
    video, caption, caption_mask, optimizer, seq_cost, random_sample = model()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(max_to_keep=100)

        # vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # for v in vs:
        #     print(v.name)
        for epoch in range(n_epoch):
            print('Epoch :', epoch)
            start = time.time()
            total_cost = 0
            for i, (feat, cap) in enumerate(loader.train_data(batch_size)):
                if i % 30 < 10:
                    sample = np.ones(max_caption_length)
                else:
                    sample = np.random.choice(2, max_caption_length)

                cap_mask = [[1 if i != 0 else 0 for i in c] for c in cap]
                
                feed = {
                    video: feat,
                    caption: cap,
                    caption_mask: cap_mask,
                    random_sample: sample
                }
                _, cost = sess.run([optimizer, seq_cost], feed_dict=feed)
                total_cost += cost
                if (i+1) % 100 == 0:
                    print('Time :', time.time() - start, 'Loss :', total_cost/(i+1))

            avg_cost = total_cost/(i+1)
            print('Time :', time.time() - start, 'Loss :', avg_cost)

            saver.save(sess, output_path + 'epoch%d-%.2f' % (epoch+1, avg_cost))

def test(testing_dir, testing_list, model_dir='./att_model/'):
    def test_data(batch_size):
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
    video, seq_output = model(test=True)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        
        path = tf.train.latest_checkpoint(model_dir)

        saver.restore(sess, path)

        loader = pickle.load(open(os.path.join(model_dir, 'loader.p'), 'rb'))

        voca = loader.voca
        inv_voca = {v: k for k, v in voca.items()}

        output = []
        
        
        for id, feat in test_data(10):
            feed = {video: feat}

            pred = seq_output.eval(feed)

            for i, row in zip(id, pred):
                seq = ''
                for word in row:
                    word = inv_voca[word]
                    if word == '<eos>':
                        break
                    if word == '<pad>':
                        word = ''
                    seq += ' ' + word
                seq = seq.strip()
                d = {'caption': seq, 'id': i}
                output.append(d)

        open('output.json', 'w').write(json.dumps(output))
