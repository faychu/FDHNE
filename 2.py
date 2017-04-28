import nltk
import scipy
import re
import sys
import langid
import string
from six.moves import cPickle as pickle
import collections
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
print('hello world!')
with open('tfidf.pickle', 'rb') as f:
    tfidf_dic = pickle.load(f)
print tfidf_dic[0].shape