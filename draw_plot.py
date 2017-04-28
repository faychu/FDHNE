# coding:utf-8
__author__ = 'fay'
import nltk
import scipy
import re
import sys
import langid
import string
import collections
import matplotlib
matplotlib.use('Agg') # 实现Matplotlib绘图并保存图像但不显示图形的方法
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

with open('count_word.pickle', 'rb') as f:
    data = pickle.load(f)

# print(data.most_common(2300))
Y = data['allen']
print(Y)
# Y.sort(reverse = True)
# print(Y[999])
# for i in range(1000,6000,100):
#     print(str(i)+':'+str(Y[i]))
# X = range(len(data))
# print(len(X))
#
# plt.scatter(X, Y)
# plt.savefig('1.png')