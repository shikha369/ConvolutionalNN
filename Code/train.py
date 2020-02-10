
import argparse
import numpy as np
import trainCalled

FLAGS = argparse.ArgumentParser(
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
FLAGS.add_argument('-lr', '--lr',type=float,
                 help='input learning rate')
FLAGS.add_argument('-batch_size', '--batch_size',type=int,
                 help='batch size..valid values are 1 and multiples of 5')
FLAGS.add_argument('-init', '--init',type=int,
                 help='Enter 1 for Xavier and 2 for He')
FLAGS.add_argument('-save_dir', '--save_dir',
                 help='path to save your model')

if __name__ == '__main__':
    args = FLAGS.parse_args()
    print args.lr
    trainCalled.called(args)
