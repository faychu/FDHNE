# coding:utf-8
__author__ = 'fay'
from six.moves import cPickle as pickle
import nltk
import sys
import string
import collections
reload(sys)
sys.setdefaultencoding('utf-8')
a = 'Jamie Foxx ft. Yung Joc & T-Pain - Blame It (Remix)'+' abdg et an apple my weird character \x96'

a = unicode(a, errors = 'ignore')
delset = string.punctuation + string.digits + '+――！，。？、~@#￥%……&*（）' + '.!/_,$%^*(+"'
translate_table = dict((ord(char), u'') for char in delset)
a = a.translate(translate_table)
a = a.lower()
stopwords = nltk.corpus.stopwords.words('english') # 停用词列表
stemmer = nltk.PorterStemmer()  # 波特词干提取器
tokenizer = nltk.RegexpTokenizer('\w+')
a = tokenizer.tokenize(a)
a = [stemmer.stem(word) for word in a if not word in stopwords]
with open('1.txt', 'w') as f:
    f.write(str(a))
# print(a)

with open('txt.pickle', 'rb') as f:
    data = pickle.load(f)

print(data[3327302])
a = collections.Counter([1,1,3,4])
print(type(a))