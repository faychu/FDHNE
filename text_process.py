# coding:utf-8
__author__ = 'fay'
import nltk
import scipy
import re
import sys
import langid
import string
from six.moves import cPickle as pickle
import collections
import matplotlib
matplotlib.use('Agg') # 实现Matplotlib绘图并保存图像但不显示图形的方法
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

reload(sys)
sys.setdefaultencoding('utf-8')


# 顺序：
# 1、检测语种
# 2、去掉数字和标点符号
# 3、小写
# 4、分词
# 5、去停用词
# 6、提取词干
def raw2dic(filelist):
    dic1 = {}
    delset = string.punctuation + string.digits + '+――！，。？、~@#￥%……&*（）' + '.!/_,$%^*(+"'
    stopwords = nltk.corpus.stopwords.words('english')  # 停用词列表
    stemmer = nltk.PorterStemmer()  # 波特词干提取器
    # newfile = open('text_id2str_all.txt', 'w')
    for file in filelist:
        with open(file, 'rb') as f:
            next(f)
            for eachLine in f:
                if langid.classify(eachLine)[0] == 'en':  # 不处理非英文的文本
                    a = unicode(eachLine, errors='replace')  # 解码为unicode
                    translate_table = dict((ord(char), u'') for char in delset)
                    a = a.split('\t')[1].translate(translate_table)  # 去掉数字和标点符号
                    a = a.lower()  # 小写
                    tokenizer = nltk.RegexpTokenizer('\w+')
                    a = tokenizer.tokenize(a)  # 分词
                    a = [stemmer.stem(word) for word in a if not word in stopwords]
                    if len(a) >= 1:
                        dic1.setdefault(int(eachLine.split('\t')[0]), a)
                        # dic1.setdefault(int(eachLine.split('\t')[0]), ' '.join(a))
                        # newfile.write(eachLine.split('\t')[0]+'\t'+str(a)+'\n')
                        # newfile.write(eachLine.split('\t')[0]+'\t'+' '.join(a)+'\n')
    with open('text_id2list.pickle', 'wb') as f:
        pickle.dump(dic1, f, pickle.HIGHEST_PROTOCOL)

filelist = ['Textual_Features/200812/title_200812_expanded.txt',
            'Textual_Features/200812/title_200812_expanded_sa.txt',
            'Textual_Features/200812/title_200812_core.txt',
            'Textual_Features/200901/title_200901_expanded.txt',
            'Textual_Features/200901/title_200901_expanded_sa.txt',
            'Textual_Features/200901/title_200901_core.txt',
            'Textual_Features/200902/title_200902_expanded.txt',
            'Textual_Features/200902/title_200902_expanded_sa.txt',
            'Textual_Features/200902/title_200902_core.txt']

# raw2dic(filelist)

def count_word(picklefile):
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    word_list = [x for j in data.values() for x in j]
    a = collections.Counter(word_list)
    with open('count_word.pickle', 'wb') as f:
        pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)


# count_word('txt.pickle')


def generate_dictionary(countfile):
    with open(countfile, 'rb') as f:
        data = pickle.load(f)
    tuple_list = data.most_common(2000)
    dictionary = [i[0] for i in tuple_list]
    with open('dictionary.pickle', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    with open('dictionary.txt', 'w') as f:
        for i in dictionary:
            f.writelines(i+'\n')
    return dictionary

# generate_dictionary('count_word.pickle')


def refine_text(pickle_file, dictionary_list):
    refine_dic = {}
    file = open('refine_text_withvis.txt', 'w')
    with open('hlf.pickle', 'rb') as f:
        data1 = pickle.load(f)
    list1 = data1.keys()
    with open(pickle_file, 'rb') as f:
        text = pickle.load(f)
    for i in text.keys():
        a_list = text[i]
        refine_list = [item for item in a_list if item in dictionary_list]
        if len(refine_list) >= 1 and i in list1:
            refine_dic.setdefault(i, refine_list)
            file.write(str(i)+'\t'+str(refine_list)+'\n')
    with open('refine_text_withvis.pickle', 'wb') as f:
        pickle.dump(refine_dic, f, pickle.HIGHEST_PROTOCOL)
    print(len(refine_dic))
    return refine_dic

# refine_text('text_id2list.pickle', generate_dictionary('count_word.pickle'))

def Tfidf(picklefile):
    tfidf_dic = {}
    corpus = []
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    for i in range(44144):
        corpus.append(data[i])
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print weight.shape
    for i in range(44144):
        tfidf_dic.setdefault(i, weight[i, :])
    print(tfidf_dic[0].shape)
    with open('tfidf.pickle','wb') as f:
        pickle.dump(tfidf_dic, f, pickle.HIGHEST_PROTOCOL)
    with open('tfidf_word.pickle', 'wb') as f:
        pickle.dump(word, f, pickle.HIGHEST_PROTOCOL)
    # with open('weight.pickle', 'wb') as f:
    #     pickle.dump(weight, f, pickle.HIGHEST_PROTOCOL)
    np.save('tfidf_weight', weight)

# Tfidf('refine_text_id2str.pickle')
weight = np.load('tfidf_weight.npy')
print weight.shape
c = np.sum(weight, axis=1)
for i in range(44144):
    assert c[i] != 0
print(c.shape)




# list1 = data1.keys()
# with open('refine_text.pickle', 'rb') as f:
#     data2 = pickle.load(f)
# list2 = data2.keys()
# list3 = [item for item in list2 if item in list1]

# with open('video_with_vis&txt.pickle', 'wb') as f:
#     pickle.dump(list3, f, pickle.HIGHEST_PROTOCOL)
# print(len(list3))


# generate refine_text_id2str
def generate_refine_text_id2str():
    dic = {}
    with open('vid_dict.pickle', 'rb') as f:
        vid_dict = pickle.load(f)
    with open('refine_text_withvis.pickle', 'rb') as f:
        refine = pickle.load(f)
    for i in refine:
        dic.setdefault(vid_dict[i], ' '.join(refine[i]))
    assert len(dic) == 44144
    return dic


# dic = generate_refine_text_id2str()
# for i in range(10):
#     print(dic[i])
# with open('refine_text_id2str.pickle', 'wb') as f:
#     pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

with open('tfidf.pickle', 'rb') as f:
    data = pickle.load(f)
    for i in range(10):
        print(data[0])