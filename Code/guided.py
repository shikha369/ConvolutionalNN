from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches



import tensorflow as tf
import numpy as np
import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from datetime import datetime
import math
import patch_reconstruct_image
import Custom_initialisation

# Replace vanila relu to guided relu to get guided backpropagation.
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))
checkpoint_dir='/tmp/cifar10_train'
batch_size = 1
MOVING_AVERAGE_DECAY = 0.9999
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',1,
                           """...""")

def getIndex(a):
   i = np.unravel_index(a.argmax(), a.shape)
   return i


def do_me_a_favor(saver, gb_grad, target_conv_layer, target_conv_layer_grad_norm,images,labels):

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
      while step < 1 and not coord.should_stop():
          #a = np.array(target_conv_layer[0].eval())
          #print (a.shape)
          '''zero_mask = tf.zeros_like(conv3)
          max_act = tf.reduce_max(conv3)
          max_mask = tf.constant(max_act, dtype=tf.float32, shape=conv3.shape)
          conv3 = tf.select( tf.equal(conv3, max_mask), conv3, zero_mask )
          #mask = tf.assign(mask,1)

          #conv3 = tf.multiply(conv3,mask)'''

          gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad_norm])
          a = np.array(target_conv_layer_value)
          max_index = np.unravel_index(a.argmax(), a.shape)
          print(max_index)
          print (tf.count_nonzero(target_conv_layer_value))

          utils.visualize(images, target_conv_layer_value, target_conv_layer_grad_value, gb_grad_value)

          step = step + 1





    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()

    coord.join(threads, stop_grace_period_secs=10)



def _variable_on_cpu(name, shape, initializer,trainable= True):
  with tf.device('/cpu:0'):
    dtype =tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable = trainable)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, init):
  dtype =tf.float32
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

def main(args=None):
  init =1
  eval_graph = tf.Graph()
  with eval_graph.as_default():
      with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):

        images, labels = patch_reconstruct_image.inputs(-1,1)
        with tf.variable_scope('conv1') as scope:
          kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0,init=init)
          conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
          biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
          pre_activation = tf.nn.bias_add(conv, biases)
          conv1 = tf.nn.relu(pre_activation, name=scope.name)


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

          ##DO something here
          #a = np.array(conv3)
          #mask = tf.ones_like(conv3)
          mask = tf.zeros_like(conv3)
          indices = [[0, 1, 2,198]]
          #indices = tf.arg_max(conv3,0)
          values = [1.0]
          shape = [1, 6, 6, 256]
          delta = tf.SparseTensor(indices, values, shape)
          mask = mask + tf.sparse_tensor_to_dense(delta)
          conv3 = tf.multiply(conv3,mask)
          #print (mask)




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


  # FC2
        with tf.variable_scope('fc2') as scope:
          weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                          stddev=0.04, wd=0.004,init=init)
          biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
          fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)


        with tf.variable_scope('softmax_linear') as scope:
          weights = _variable_with_weight_decay('weights', [1024, 10],
                                          stddev=1/1024.0, wd=0.0, init=init)
          biases = _variable_on_cpu('biases', [10],
                              tf.constant_initializer(0.0))
          softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=softmax_linear)

        opt = tf.train.AdamOptimizer(0.001,beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cross_entropy)
        #opt_op = opt.minimize(cross_entropy)
        #opt_op.run()

        target_conv_layer = conv3
        target_conv_layer_grad = tf.gradients(cross_entropy, target_conv_layer)[0]
        gb_grad = tf.gradients(target_conv_layer, images)[0]
        target_conv_layer_grad_norm = tf.div(target_conv_layer_grad, tf.sqrt(tf.reduce_mean(tf.square(target_conv_layer_grad))) + tf.constant(1e-5))


        variable_averages = tf.train.ExponentialMovingAverage(
          MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)





        do_me_a_favor(saver, gb_grad, target_conv_layer, target_conv_layer_grad_norm,images,labels)



if __name__ == '__main__':
  tf.app.run()
