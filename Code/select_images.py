#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:59:07 2017

@author: shikha
"""
import os

data_dir = os.path.join('/tmp/cifar10_data', 'cifar-10-batches-bin')
fileread = os.path.join(data_dir, 'data_batch_5.bin')
file1 = open(fileread, 'rb')
bytes_to_read = 32*32*3 + 1

filewrite_image = os.path.join(data_dir, 'image1.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image2.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image3.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image4.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image5.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image6.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image7.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image8.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image9.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()

filewrite_image = os.path.join(data_dir, 'image10.bin')
byte = file1.read(bytes_to_read)
file2 = file(filewrite_image, 'wb')
file2.write(byte)
file2.close()



file1.close()