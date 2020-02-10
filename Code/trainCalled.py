from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re

import numpy as np
from six.moves import xrange
import tensorflow as tf
import Network
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 90000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def tower_loss(scope):
  images, labels = Network.distorted_inputs()
  logits = Network.model(images,FLAGS.init,True)
  _ = Network.loss(logits, labels)
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  for l in losses + [total_loss]:
    loss_name = re.sub('%s_[0-9]*/' % Network.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)
  return total_loss


def my_tower_loss(scope,data):
  images, labels = Network.inputs(data)
  logits = Network.model(images,FLAGS.init,False)
  _ = Network.loss(logits, labels)
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  return total_loss

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      print (g)
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    num_batches_per_epoch = (45000/
                             FLAGS.batch_size)
    lr = FLAGS.lr
    patience_Counter = 0
    patience_Param = 5
    prev_validation_loss = 500 # some high value

    opt = tf.train.AdamOptimizer(lr,beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (Network.TOWER_NAME, i)) as scope:
            loss = tower_loss(scope)
            tf.get_variable_scope().reuse_variables()
            vloss = my_tower_loss(scope,0)
            tf.get_variable_scope().reuse_variables()
            tloss = my_tower_loss(scope,1)
            tf.get_variable_scope().reuse_variables()
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    summaries.append(tf.summary.scalar('learning_rate', lr))
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    variable_averages = tf.train.ExponentialMovingAverage(
        Network.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge(summaries)

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    file_logs = open(FLAGS.train_dir+"/log_loss.txt", "w")

    for step in xrange(FLAGS.max_steps):
      _, loss_value, v_loss_value, t_loss_value = sess.run([train_op, loss, vloss, tloss])

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        format_str = ('step %d, train_loss = %.2f, valid_loss = %.2f and test_loss = %.2f')
        print (format_str % (step, loss_value
                             , v_loss_value, t_loss_value ))
        file_logs.write("Step %i, train_Loss: %f, valid_Loss: %f, test_Loss: %f\n" %(step, loss_value, v_loss_value, t_loss_value))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if (v_loss_value > prev_validation_loss and patience_Counter < patience_Param):
          patience_Counter += 1
      elif (v_loss_value <= prev_validation_loss and patience_Counter < patience_Param):
          patience_Counter = 0
      else :
          FLAGS.max_steps = step

    file_logs.close()


def called(args):
  tf.app.flags.DEFINE_integer('batch_size', args.batch_size,
                            """Number of images to process in a batch.""")
  tf.app.flags.DEFINE_float('lr', args.lr,
                            """learning rate.""")
  tf.app.flags.DEFINE_integer('init', args.init,
                            """weight initialiser.""")

  Network.download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  Network.create_validation_set()
  train()


if __name__ == '__main__':
  tf.app.run()

