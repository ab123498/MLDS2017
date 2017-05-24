import tensorflow as tf
import ops

class GAN:
    '''
    OPTIONS
    z_dim : Noise dimension 100
    t_dim : Text feature dimension 256
    image_size : Image Dimension 64
    gf_dim : Number of conv in the first layer generator 64
    df_dim : Number of conv in the first layer discriminator 64
    gfc_dim : Dimension of gen untis for for fully connected layer 1024
    caption_vector_length : Caption Vector Length 2400
    batch_size : Batch Size 64
    '''
    def __init__(self, options):
        self.options = options


    def build_model(self):
        img_size = self.options['image_size']
        t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
        t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])

        fake_image = self.generator(t_z, t_real_caption)
        
        disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)
        disc_wrong_image, disc_wrong_image_logits   = self.discriminator(t_wrong_image, t_real_caption, reuse = True)
        disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image)))
        
        d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
        d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits, labels=tf.zeros_like(disc_wrong_image)))
        d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))

        d_loss = d_loss1 + d_loss2 + d_loss3

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        input_tensors = {
            't_real_image' : t_real_image,
            't_wrong_image' : t_wrong_image,
            't_real_caption' : t_real_caption,
            't_z' : t_z
        }

        variables = {
            'd_vars' : d_vars,
            'g_vars' : g_vars
        }

        loss = {
            'g_loss' : g_loss,
            'd_loss' : d_loss
        }

        outputs = {
            'generator' : fake_image
        }

        checks = {
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3' : d_loss3,
            'disc_real_image_logits' : disc_real_image_logits,
            'disc_wrong_image_logits' : disc_wrong_image,
            'disc_fake_image_logits' : disc_fake_image_logits
        }
        
        return input_tensors, variables, loss, outputs, checks

    def build_generator(self):
        img_size = self.options['image_size']
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        fake_image = self.generator(t_z, t_real_caption)
        
        input_tensors = {
            't_real_caption' : t_real_caption,
            't_z' : t_z
        }
        
        outputs = {
            'generator' : fake_image
        }

        return input_tensors, outputs

    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self, t_z, t_text_embedding):
        batch_norm = tf.layers.batch_normalization
        
        s = self.options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        
        reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
        z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
        h0 = batch_norm(h0, name='g_bn0')
        h0 = ops.lrelu(h0)
        
        h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
        h1 = batch_norm(h1, name='g_bn1')
        h1 = ops.lrelu(h1)
        
        h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
        h2 = batch_norm(h2, name='g_bn2')
        h2 = ops.lrelu(h2)
        
        h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
        h3 = batch_norm(h3, name='g_bn3')
        h3 = ops.lrelu(h3)
        
        h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
        
        return (tf.tanh(h4)/2. + 0.5)

    # DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def discriminator(self, image, t_text_embedding, reuse=False):
        batch_norm = tf.layers.batch_normalization
        
        with tf.variable_scope('discriminator', reuse=reuse):
            h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32

            conv_h1 = ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv')
            bn_h1 = batch_norm(conv_h1, name='d_bn_h1', reuse=reuse)
            h1 = ops.lrelu(bn_h1) #16

            conv_h2 = ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv')
            bn_h2 = batch_norm(conv_h2, name='d_bn_h2', reuse=reuse)
            h2 = ops.lrelu(bn_h2) #16

            conv_h3 = ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv')
            bn_h3 = batch_norm(conv_h3, name='d_bn_h3', reuse=reuse)
            h3 = ops.lrelu(bn_h3) #16
            
            # ADD TEXT EMBEDDING TO THE NETWORK
            reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
            tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
            
            h3_concat = tf.concat([h3, tiled_embeddings], 3, name='h3_concat')
            h3_bn = batch_norm(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'), name='d_bn_h3_new', reuse=reuse)
            h3_new = ops.lrelu(h3_bn) #4
            
            h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
            
            return tf.nn.sigmoid(h4), h4
