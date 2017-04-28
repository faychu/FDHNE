__author__ = 'fay'
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
import csv
import nltk
import re
import string
import sys
import langid
import codecs

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_files

reload(sys)
sys.setdefaultencoding('utf-8')
"""
function mkdir
功能：新建一个文件夹
输入：
    path：文件夹地址,例如:'dealcontent_Twitter'或'dealcontent_Twitter/'
输出：无
"""
def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print path + ' 文件创建成功'
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path + ' 文件目录已存在'
        return False

"""
function recFileName
功能：得到文件夹中文件名组成的列表
输入：
    file_dir：文件夹地址,例如:'dealcontent_Twitter'或'dealcontent_Twitter/'
输出：
    L：文件名组成的列表
"""
def recFileName(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
        #     if os.path.splitext(file)[1] == '.jpeg':
        #     L.append(os.path.join(root, file))
            L.append(file)
    print L
    return L

"""
function getContentOfCsv
功能：取出csv文件中的第三列内容
输入：
    files_in：文件名列表,例如:
输出：
    将结果写入新的文件中
"""
def getContentOfCsv(files_in,fin_path,fout_path):
    # mkdir('Twitter/')

    for file_in in files_in:
        with open(fin_path+file_in, 'r') as f_in:
            with open(fout_path+file_in, 'w') as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                for tweet_id,created_at,text in reader:
                    # writer.writerow(text)
                    if reader.line_num == 1:
                        continue
                    f_out.write(text+'\n')
    f_in.close()
    f_out.close()
"""
function dealContentOfTwitter
功能：处理Twitter用户的tweet内容，包括检测语种、去除无关内容、提取#内容、删除数字标点、分词和stemming
输入：
    files_in：文件名列表,例如:['10006012_tweets.csv','10006052_tweets.csv']
输出：
    将结果写入新的文件中
"""
def dealContentOfTwitter(files_in,finContent_path,fout_path,finfull_stopwords):
    # files_in = ['10006012_tweets.csv','10006052_tweets.csv']  #
    # fout_path='twitter_deal/'
    # finContent_path = fout_path + 'twitter_content/'
    fout_path1=fout_path+'twitter_dealcontent/'#存放预处理后的word文档的文件夹
    fout_path2=fout_path+'twitter_findtag/'#存放’#word‘的文档的文件夹
    mkdir(fout_path)
    mkdir(fout_path1)  #储存处理后文件的文件夹
    mkdir(fout_path2)  # 储存处理后文件的文件夹
    corpus_sum = []#存储所有文件的结果
    stopwords_fin = open(finfull_stopwords, 'r')#'data/stop-words-list/stop_words_eng.txt'
    # stopwords_fin = codecs.open('stop-words-list/stop_words_eng.txt', 'r','utf-8')
    recordLanguage_fout=open(fout_path+'result_recordLanguage.txt','w')
    stopwordslist = stopwords_fin.read()
    file_num=0
    for file in files_in:
        file_num=file_num+1
        print '待处理的file名'
        print file
        print '已完成的文件数'
        print file_num-1
        data_fin = open(finContent_path + file, 'r')
        dealcontent_fout = open(fout_path1 + file, 'w')#写入预处理后的结果
        findtag_fout = open(fout_path2 + file, 'w')  # 写入预处理后的结果
        corpus = []#存储一个文件的结果
        num_line=0
        num_line_noEnglish=0
        # proportion_language=0
        for line in data_fin.readlines():
            num_line=num_line+1
            # print ("The num_line is %f" % (num_line))
            print '待处理的line'
            print line
            # 1.检测语种
            lineTuple = langid.classify(line)  # 输出结果是一个二元组，第一项表示该文本所属的语系，第二项表示该文本中属于第一项中语系的所占比例
            if lineTuple[0] != "en":  # 如果该行语言大部分为非英文，则不进行任何处理
                num_line_noEnglish=num_line_noEnglish+1
                # print ("The num_line_noEnglish is %f" % (num_line_noEnglish))
                continue
            # 2.删除RT和@和http:/与https:/
            line_temp  = re.sub(r'(http)\S+|RT|@\S+', '', line)#\S为匹配任何非空白字符。|(https:/)\S+、(http://t.co/)\s+|(https://t.co/)\s+| post a blog!
            # 3.提取#关键词
            findlist=re.findall('#(\w+)',line)#只返回（）内的内容列表
            for item in findlist:
                findtag_fout.write(item + '\n')
            # 4.删除数字和标点符号
            delset = string.punctuation + string.digits + '+――！，。？、~@#￥%……&*（）' + '.!/_,$%^*(+"'  # 所有标点
            line_temp = line_temp.translate(None, delset)  # 不翻译，只删除
            # print line_temp
            # 5.分词、小写、去停用词和词干提取
            corpus_stem = []  # 储存处理后和提取词干的corpus
            tokenizer = nltk.RegexpTokenizer('\w+')
            stemmer = nltk.PorterStemmer()  # 波特词干提取器
            # corpus2 = nltk.regexp_tokenize(corpus2,'\w+')#分词
            line_temp = tokenizer.tokenize(line_temp)  # 分词
            print '分词后的结果为'
            print line_temp #输出分词后的line

            for word in line_temp:
                word_temp = word.decode('utf-8', 'ignore').encode('utf-8', 'ignore')#处理编码错误
                if word_temp=='':  #'\uFFFD'# 错误处@lisalaposte you're dealing with int
                    print word
                    continue
                word_temp = word_temp.lower()  # 小写
                # print word_temp
                if word_temp in stopwordslist:  # 去停用词
                    # print  word_temp
                    # line_temp.remove(word)
                    continue
                else:
                    try:
                        word_temp = stemmer.stem(word_temp)  # 词干提取
                    except UnicodeDecodeError:
                        print 'UnicodeDecodeError'
                        print word_temp
                        continue
                    dealcontent_fout.write(word_temp+'\n')  # 将处理后的单个结果写入新文件中,以空格分隔
                    corpus_stem.append(word_temp)
                    # print corpus_stem
            print 'stemming后的结果为'
            print corpus_stem #输出处理后的line
        #     corpus.extend(corpus_stem)
        # corpus_sum.append(corpus)
        # print corpus
        findtag_fout.close()
        data_fin.close()
        dealcontent_fout.close()
        # print ("The sum num_line is %f" % (num_line))
        # print ("The sum num_line_noEnglish is %f" % (num_line_noEnglish))
        proportion_language = float(num_line_noEnglish) / num_line
        print("The proportion_language is %f" % (proportion_language))
        recordLanguage_fout.write(file + '      ')  # 记录用户的非英语line的出现率
        recordLanguage_fout.write(str(num_line_noEnglish))  # 记录用户的非英语line的num
        recordLanguage_fout.write( '      ')
        recordLanguage_fout.write(str(num_line))  # 记录用户的总line的num
        recordLanguage_fout.write( '      ')
        recordLanguage_fout.write(str(proportion_language))  # 记录用户的非英语line的出现率
        recordLanguage_fout.write('\n')  # 记录用户的非英语line的出现率
    # print corpus_sum
    stopwords_fin.close()
    recordLanguage_fout.close()

"""
function dealContentOfTwitter
功能：将多个用户的关键词结果合成一个大矩阵
输入：
    files_in：文件名列表,例如:['10006012_tweets.csv','10006052_tweets.csv']
输出：
    将结果写入新的文件中
"""
def csv_content(files_in,filesIn_Path) :
    # filesIn_Path = 'Twitter/'
    corpus = []  #存取100份文档的分词结果,列表
    for file in files_in :
        fname = filesIn_Path + file
        f = open(fname,'r+')
        content = f.read()
        f.close()
        corpus.append(content)
    print corpus
    print len(corpus)
    return corpus


def ProProcessing():
    finfull_stopwords = 'D:/A_bigdata/deal_python/lda_test/twitter/data_fin/stop-words-list/stop_words_eng.txt'
    finPathFormer = 'D:/A_bigdata/deal_python/lda_test/twitter/data_fin/twitter/'
    finPathEnd='6820/'#处理的文件夹名字
    fin_path=finPathFormer+finPathEnd#输入文件夹
    files_in= recFileName(fin_path)
    foutPathFormer = 'D:/A_bigdata/deal_python/lda_test/twitter/twitter_deal/'
    foutPathEnd=finPathEnd
    fout_path=foutPathFormer+foutPathEnd#输出文件夹
    foutConent_path = fout_path+'twitter_content/'#提取内容后的tweet的存放地址
    #创建缺失文件夹
    mkdir(foutPathFormer)
    mkdir(fout_path)
    mkdir(foutConent_path)
    #得到第三列内容
    getContentOfCsv(files_in, fin_path, foutConent_path)#得到tweet内容
    print '以获取文件的第三列内容'
    finContent_path=foutConent_path
    # files_in = ['12418482_tweets.csv']
    #处理tweet内容
    dealContentOfTwitter(files_in, finContent_path, fout_path,finfull_stopwords)#处理内容


def LdaTest():
    # fin_path = 'data_fin/twitter/764/'
    fin_path='D:/A_bigdata/deal_python/lda_test/twitter/twitter_deal/764/20_tag/'
    files_in= recFileName(fin_path)
    # filesIn_Path='twitter_deal/764/twitter_findtag/'
    filesIn_Path='D:/A_bigdata/deal_python/lda_test/twitter/twitter_deal/764/20_tag/'
    corpus=csv_content(files_in,filesIn_Path)#将多个文件合成一个一个大矩阵
    # print corpus[1]
    vectorizer = CountVectorizer(binary = False, decode_error = 'ignore',stop_words = None)#utf-8是默认
    X = vectorizer.fit_transform(corpus)#计算个词语出现的次数
    print X
    # print (vectorizer.get_feature_names())#查看特征词
    print (X.toarray())#通过toarray()可看到词频矩阵的结果。
    # print  vectorizer.get_stop_words()#查看内置停用词


    import lda
    import lda.datasets

    model = lda.LDA(n_topics=3, n_iter=500, random_state=1)
    model.fit_transform(X)
    topic_word = model.topic_word_  # model.components_ also works
    print(topic_word[:, :3])
    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_

    # 主题包含的top10 Word
    # 每个用户所属的主题

    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))
    # 输出前12篇文章最可能的Topic
    label = []
    for n in range(18):
        print(doc_topic[n])
        topic_most_pr = doc_topic[n].argmax()
        label.append(topic_most_pr)
        print("doc: {} topic: {}".format(n, topic_most_pr))


def get_functime():
    import time
    start = time.clock()

    # 当中是你的程序
    # ProProcessing()
    LdaTest()


    elapsed = (time.clock() - start)
    print("Time used:", elapsed)



get_functime()

