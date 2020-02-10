
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from datetime import datetime
import numpy as np
import tensorflow as tf
import Network
import patch_reconstruct_image
checkpoint_dir='/tmp/cifar10_train'
eval_dir='/tmp/cifar10_fm'
batch_size = 1
MOVING_AVERAGE_DECAY = 0.9999
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',1,
                           """...""")
def eval_once(saver, summary_writer, conv1, summary_op):
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

      step = 0
      num_iter = 1
      while step < num_iter and not coord.should_stop():
        a = sess.run([conv1])
        a = np.array(a)
        filters = a[0][0]
        filters = tf.transpose(filters, (2, 0, 1))
        print (a.shape)
        print (filters.shape)

        filter_no = 0
        while filter_no < 64 :

          img = filters[filter_no]
          x_min = tf.reduce_min(img)
          x_max = tf.reduce_max(img)

          img = (img - x_min) / (x_max - x_min)
          img = img.eval()
          fig,ax = plt.subplots(1)
          ax.imshow(img,cmap='gray')
          plt.show()
          filter_no = filter_no + 1
        step = step + 1

    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()

    coord.join(threads, stop_grace_period_secs=10)


def get_conv1 (images):
  with tf.variable_scope('conv1') as scope:
    kernel = Network._variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0,init=1)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = Network._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  return conv1


def main(args=None):

  with tf.Graph().as_default() as g:

    images, labels = patch_reconstruct_image.inputs(-1,1)
    conv1 = get_conv1(images)
    variable_averages = tf.train.ExponentialMovingAverage(
        Network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(eval_dir, g)
    eval_once(saver, summary_writer, conv1, summary_op)


if __name__ == '__main__':
  tf.app.run()
