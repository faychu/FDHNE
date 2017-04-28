# coding: utf-8
from six.moves import cPickle as pickle
import collections
import numpy as np
import random
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')  # 实现Matplotlib绘图并保存图像但不显示图形的方法
import matplotlib.pyplot as plt
import scipy as sk

with open('nodeFeaturesNUS.txt', 'r') as f:
    for i in range(40217):
        next(f)
    for eachLine in f:
        l = eachLine.strip().split(' ')
        img_name = l[0]

