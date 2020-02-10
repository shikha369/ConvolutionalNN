
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import urllib
import tensorflow as tf

import Utils
from math import sqrt

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45000
NUM_EXAMPLES_PER_EPOCH_FOR_VAL_EVAL = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
data_dir='/tmp/cifar10_data/cifar-10-batches-bin'

def inputs(eval_data):
  if not data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
  images, labels = Utils.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  return images, labels

def inputs(eval_data, batch_size):
  if eval_data == -1:
    '''filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(0, 5)]'''

    filenames = [os.path.join(data_dir, 'image1.bin')]
    print (filenames)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  elif eval_data ==0:
    filenames = [os.path.join(data_dir, 'valid_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_VAL_EVAL
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)


  filename_queue = tf.train.string_input_producer(filenames)

  read_input = Utils.read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)
  float_image = resized_image/255.0

  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  return  Utils._generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

