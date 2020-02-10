
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import Utils
import Custom_initialisation
from math import sqrt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
IMAGE_SIZE = Utils.IMAGE_SIZE
NUM_CLASSES = Utils.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = Utils.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = Utils.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
MOVING_AVERAGE_DECAY = 0.9999
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def create_validation_set():
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    fileread = os.path.join(data_dir, 'data_batch_5.bin')
    filewrite_valid = os.path.join(data_dir, 'valid_batch.bin')
    filewrite_data = os.path.join(data_dir, 'data_batch_0.bin')
    file1 = open(fileread, 'rb')
    bytes_to_read = (32*32*3 + 1) * 5000
    byte = file1.read(bytes_to_read)
    file2 = file(filewrite_valid, 'wb')
    file2.write(byte)
    byte = file1.read(bytes_to_read)
    file3 = file(filewrite_data, 'wb')
    file3.write(byte)
    file1.close()
    file2.close()
    file3.close()



def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def image_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer,trainable= True):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable = trainable)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, init):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  if init == 1:
      mode = 'FAN_AVG'  # Xavier
  if init == 2:
      mode = 'FAN_IN'   # He
  var = _variable_on_cpu(
      name,
      shape,
      Custom_initialisation.Cust_initializer(uniform=False,seed = 1234, dtype=dtype,mode=mode))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = Utils.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = Utils.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def model(images,init,is_Training):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    grid = put_kernels_on_grid (kernel,pad=1)
    tf.summary.image('conv1/features', grid, max_outputs=1)
    conv = tf.nn.conv2d(images, kernel/tf.norm(kernel), [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)


  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv4)


  # pool3
  pool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # FC1
  with tf.variable_scope('fc1') as scope:
    reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004,init=init)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc1)

  # FC2
  with tf.variable_scope('fc2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                          stddev=0.04, wd=0.004,init=init)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
    _activation_summary(fc2)

  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                          stddev=1/1024.0, wd=0.0, init=init)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

  #to remove BN layer, just comment BNoutput and uncomment ret. softmax_linear
    BN_output=batch_norm_wrapper(softmax_linear,is_Training=is_Training)
  return BN_output
  #return softmax_linear

def batch_norm_wrapper(inputs, is_Training, decay = 0.999):
    epsilon = 0.001

    scale = tf.get_variable(name = 'scale',shape=[inputs.get_shape()[-1]],initializer = tf.constant_initializer(1.0))
    beta = tf.get_variable(name ='beta',shape=[inputs.get_shape()[-1]],initializer = tf.constant_initializer(0))
    pop_mean = tf.get_variable(name ='pop_mean',shape=[inputs.get_shape()[-1]],trainable=False,initializer = tf.constant_initializer(0))
    pop_var = tf.get_variable(name ='pop_var',shape=[inputs.get_shape()[-1]],trainable=False,initializer = tf.constant_initializer(1))

    if is_Training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  lr = FLAGS.lr

  tf.summary.scalar('learning_rate', lr)
  loss_averages_op = _add_loss_summaries(total_loss)
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr,beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    grads = opt.compute_gradients(total_loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')


  return train_op


def put_kernels_on_grid (kernel, pad = 1):

    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))
    x6 = tf.transpose(x5, (2, 1, 3, 0))
    x7 = tf.transpose(x6, (3, 0, 1, 2))
    return x7




def download_and_extract():
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
