import tensorflow as tf
import numpy as np
import model_ref as model
import argparse
import pickle
from data_loader import Loader
import shutil
import os
import scipy.misc

def main():
    z_dim = 100
    t_dim = 256
    image_size = 64
    gf_dim = 64
    df_dim = 64
    gfc_dim = 1024
    caption_vector_length = 2400
    batch_size = 64

    options = {
        'z_dim': z_dim,
        't_dim': t_dim,
        'image_size': image_size,
        'gf_dim': gf_dim,
        'df_dim': df_dim,
        'gfc_dim': gfc_dim,
        'caption_vector_length': caption_vector_length,
        'batch_size': batch_size
    }

    epochs = 300
    learning_rate = 0.0002
    beta1 = 0.5

    gan = model.GAN(options)
    input_tensors, variables, loss, outputs, checks = gan.build_model()
    
    d_optim = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    saver = tf.train.Saver()
    # if args.resume_model:
    #     saver.restore(sess, args.resume_model)
    
    loader = Loader()
    for i in range(epochs):
        batch_no = 0
        for real_images, wrong_images, captions in loader.train_data(batch_size):
            real_images /= 255.0
            wrong_images /= 255.0
            captions = captions[:, :caption_vector_length]

            z_noise = np.random.normal(0, 1, [batch_size, z_dim])

            # DISCR UPDATE
            check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
            _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_wrong_image'] : wrong_images,
                    input_tensors['t_real_caption'] : captions,
                    input_tensors['t_z'] : z_noise,
                })

            # GEN UPDATE
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_wrong_image'] : wrong_images,
                    input_tensors['t_real_caption'] : captions,
                    input_tensors['t_z'] : z_noise,
                })

            # GEN UPDATE TWICE, to make sure d_loss does not go to 0
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                feed_dict = {
                    input_tensors['t_real_image'] : real_images,
                    input_tensors['t_wrong_image'] : wrong_images,
                    input_tensors['t_real_caption'] : captions,
                    input_tensors['t_z'] : z_noise,
                })
            
            print(i, batch_no, d_loss, g_loss)
            
            with open('loss_log', 'a+') as out:
                out.write('%s,%s,%s,%s\n' % (i, batch_no, d_loss, g_loss))

            batch_no += 1
            if (batch_no % 30) == 0:
                print ("Saving Images, Model")
                save_for_vis(real_images, gen)
                save_path = saver.save(sess, "models/latest_model_temp.ckpt")

        if i%5 == 0:
            save_path = saver.save(sess, "models/model_epoch_%d.ckpt" % i)

def save_for_vis(real_images, fake_images):
    save_dir = './test_sample'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for i, (real, fake) in enumerate(zip(real_images, fake_images)):
        if i >= 10:
            break
        scipy.misc.imsave(os.path.join(save_dir, '%d.jpg' % i), real)
        scipy.misc.imsave(os.path.join(save_dir, '%d_fake.jpg' % i), fake)


if __name__ == '__main__':
    model_dir = './models'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    main()