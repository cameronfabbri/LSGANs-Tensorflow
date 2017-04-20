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
import argparse
import data_ops

if __name__ == '__main__':

   parser = argparse.ArgumentParser()

   parser.add_argument('--BATCH_SIZE', required=False,type=int,default=32,help='Batch size to use')
   parser.add_argument('--DATA_DIR', required=True,type=str,help='Directory containing images')
   parser.add_argument('--DATASET', required=True,type=str,help='Name of the dataset')
   a = parser.parse_args()

   DATA_DIR = a.DATA_DIR
   DATASET  = a.DATASET
   BATCH_SIZE = a.BATCH_SIZE


   checkpoint_dir = 'checkpoints/'+DATASET+'/'

   try:os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(checkpoint_dir+'images/')
   except: pass
   
   images_dir = checkpoint_dir+'images/'
   
   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1024), name='z')

   train_images_list = data_ops.loadCeleba(DATA_DIR, DATASET)
   filename_queue    = tf.train.string_input_producer(train_images_list)
   real_images       = data_ops.read_input_queue(filename_queue, BATCH_SIZE)

   # generated images
   gen_images = netG(z, BATCH_SIZE)

   # get the output from D on the real and fake data
   errD_real = tf.reduce_mean(netD(real_images, BATCH_SIZE))
   errD_fake = tf.reduce_mean(netD(gen_images, BATCH_SIZE, reuse=True))

   errD = 0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake))
   errG = 0.5*(tf.square(errD_fake - 1))

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   #tf.summary.image('real_images', real_images, max_outputs=BATCH_SIZE)
   #tf.summary.image('generated_images', gen_images, max_outputs=BATCH_SIZE)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init  = tf.global_variables_initializer()
   
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(checkpoint_dir+'/logs/', graph=tf.get_default_graph())

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

   num_train = len(train_images_list)

   while True:
      
      start = time.time()
      epoch_num = step/(num_train/BATCH_SIZE)
      batch_z = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 1024]).astype(np.float32)
      sess.run(D_train_op, feed_dict={z:batch_z})

      batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 1024]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z:batch_z})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_real, D_fake, D_loss, G_loss, summary = sess.run([errD_real, errD_fake, errD, errG, merged_summary_op], feed_dict={z:batch_z})
      summary_writer.add_summary(summary, step)

      if step%10 == 0:
         print 'epoch_num:',epoch_num,'step:',step,'D_real:',D_real,'D_fake:',D_fake,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%1000 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 1024]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         data_ops.saveImage(gen_imgs[0], step, images_dir)
         print 'Done saving'

