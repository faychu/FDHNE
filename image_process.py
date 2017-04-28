# coding:utf-8
from six.moves import cPickle as pickle
import collections
import numpy as np
import random
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')  # 实现Matplotlib绘图并保存图像但不显示图形的方法
import matplotlib.pyplot as plt
import scipy as sk

image_raw_data = tf.gfile.FastGFile('0002_174174086.jpg','r').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    resized = tf.image.resize_images(img_data,[200,200])
    plt.imshow(resized.eval())
    plt.show()
    # print(img_data.eval())