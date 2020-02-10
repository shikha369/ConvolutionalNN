
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math

import numpy as np
import tensorflow as tf
import argparse
import Network
FLAGS_B = argparse.ArgumentParser(
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
FLAGS_B.add_argument('-batch_size', '--batch_size',type=int,
                 help='batch size..valid values are 1 and multiples of 5')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size',0,
                           """...""")


def evaluate(saver, summary_writer, top_k_op,pred, summary_op):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    coord = tf.train.Coordinator()

    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      print (num_iter)
      true_count = 0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      while step < num_iter and not coord.should_stop():
        predictions,predicted = sess.run([top_k_op,pred])
        #print (predicted) --- Uncomment to get the predicted labels
        true_count += np.sum(predictions)
        step += 1
      accuracy = true_count / total_sample_count
      print('Accuracy on Test = %.3f' % (accuracy))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='accuracy', simple_value=accuracy)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()

    coord.join(threads, stop_grace_period_secs=10)


def evaluate_graph():

  with tf.Graph().as_default() as g:
    images, labels = Network.inputs(1)
    logits = Network.model(images,1,False)
    top_one_op = tf.nn.in_top_k(logits, labels, 1)
    pred = tf.arg_max(logits,1)

    variable_averages = tf.train.ExponentialMovingAverage(
        Network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    evaluate(saver, summary_writer, top_one_op,pred, summary_op)


def main(args):
  args = FLAGS_B.parse_args()
  print (args.batch_size)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  FLAGS.batch_size = args.batch_size
  Network.download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  evaluate_graph()


if __name__ == '__main__':
  tf.app.run()
