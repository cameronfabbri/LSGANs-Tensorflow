import scipy.misc as misc
import time
import tensorflow as tf
from architecture import netD, netG
import numpy as np
import random
import ntpath
import sys
import cv2
import os
from skimage import color

import data_ops
import config

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(checkpoint_dir):

   batch_size     = config.batch_size
   data_dir       = config.data_dir
   dataset        = config.dataset

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(batch_size, 100), name='z')

   train_images_list = data_ops.loadCeleba(data_dir)
   filename_queue    = tf.train.string_input_producer(train_images_list)
   real_images       = data_ops.read_input_queue(filename_queue)

   # generated images
   gen_images = netG(z, batch_size)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, batch_size)
   errD_fake = netD(gen_images, batch_size, reuse=True) # gotta pass reuse=True to reuse weights

   # cost functions
   a = 1
   b = 1
   c = 1
   errD = tf.reduce_mean((errD_real-b)**2 - (errD_fake-a)**2)
   errG = tf.reduce_mean((errD_fake-c)**2)

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   tf.summary.image('real_images', real_images, max_outputs=batch_size)
   tf.summary.image('generated_images', gen_images, max_outputs=batch_size)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step)

   saver = tf.train.Saver(max_to_keep=1)
   init  = tf.global_variables_initializer()
   #init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(checkpoint_dir+dataset+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   while True:
      
      start = time.time()

      # get the discriminator properly trained at the start
      #if step < 25 or step % 500 == 0:
      #   n_critic = 100
      #else: n_critic = 5

      # train the discriminator for 5 or 25 runs
      #for critic_itr in range(n_critic):
      batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 1024]).astype(np.float32)
      sess.run(D_train_op, feed_dict={z:batch_z})

      # now train the generator once! use normal distribution, not uniform!!
      batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 1024]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z:batch_z})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%1000 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.uniform(-1.0, 1.0, size=[batch_size, 1024]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         data_ops.saveImage(gen_imgs[0], step)
         print 'Done saving'



if __name__ == '__main__':

   checkpoint_dir = config.checkpoint_dir
   learning_rate  = config.learning_rate
   batch_size     = config.batch_size
   data_dir       = config.data_dir
   dataset        = config.dataset
   if checkpoint_dir[-1] is not '/': checkpoint_dir+='/'
   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(checkpoint_dir+dataset)
   except: pass
   try: os.mkdir('images/')
   except: pass
   try: os.mkdir('images/'+dataset)
   except: pass
   
   checkpoint_dir = checkpoint_dir+dataset+'/'
   
   buildAndTrain(checkpoint_dir)

