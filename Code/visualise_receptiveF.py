from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from datetime import datetime
import math

import numpy as np
import tensorflow as tf
import Network
import patch_reconstruct_image
checkpoint_dir='/tmp/cifar10_train'
eval_dir='/tmp/cifar10_patch'
batch_size = 1
MOVING_AVERAGE_DECAY = 0.9999
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',1,
                           """...""")
def get_receptiveField_max_active_neuron(saver, summary_writer, conv3,images,labels, summary_op):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    print("Restored checkpoint from:", global_step)

    coord = tf.train.Coordinator()

    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_images_to_pass = 300 # no. of images you are looking at
      num_iter = int(math.ceil(num_images_to_pass / batch_size))
      print (num_iter)
      step = 0
      index = np.zeros(shape=(10000,4))

      while step < num_iter and not coord.should_stop():
        a,label,image = sess.run([conv3,labels,images])
        a = np.array(a)
        max_index = np.unravel_index(a.argmax(), a.shape)
        print (step)
        y = max_index[1]
        x = max_index[2]
        d = max_index[3]
        l = label
        index[step,0]= x
        index[step,1]= y
        index[step,2]= d
        index[step,3]= l

        '''
         Given the dimension of each layer;the endpoints of bounding box are calculated manually
        '''
        x_min = 4*x - 7
        x_max = 4*x + 10
        y_min = 4*y - 7
        y_max = 4*y + 10
        if x_min<0 :
            x_min = 0
        if y_min < 0 :
            y_min = 0
        if x_max > 23:
            x_max = 23
        if y_max > 23:
            y_max = 23

        if l==8: #class you are interested in
          img = image[0]
          fig,ax = plt.subplots(1)
          ax.imshow(img)
          rect = patches.Rectangle((x_min,y_min),(x_max-x_min+1),(y_max-y_min+1),linewidth=1,edgecolor='r',facecolor='none')
          ax.add_patch(rect)
          plt.show()
        step = step + 1

    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()

    coord.join(threads, stop_grace_period_secs=10)


def get_conv3_actiavtions (images,init,is_Training):
  with tf.variable_scope('conv1') as scope:
    kernel = Network._variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = Network._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = Network._variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = Network._variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel =Network. _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = Network._variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
  return conv3


def main(args=None):

  with tf.Graph().as_default() as g:

    images, labels = patch_reconstruct_image.inputs(-1,1)
    conv3 = get_conv3_actiavtions(images,1,False)

    variable_averages = tf.train.ExponentialMovingAverage(
        Network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)
    get_receptiveField_max_active_neuron(saver, summary_writer, conv3,images,labels, summary_op)


if __name__ == '__main__':
  tf.app.run()
