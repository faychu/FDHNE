__author__ = 'fay'
import collections
import math
import os
import re
import tarfile
from contextlib import closing
import random
from six.moves import cPickle as pickle
import numpy as np
import nltk

def load_hlf(folders):
    hlf_dic = {}
    for folder in folders:
        for video in os.listdir(os.path.join('HLF346_ExpandedData', folder)):
            video_file = os.path.join('HLF346_ExpandedData', folder, video)
            sum_list = np.zeros(346, float)
            with open(video_file, 'r') as f:
                # line_num = 0
                for eachLine in f:
                    sum_list += [float(i.split(':')[1]) for i in eachLine.strip().split()[1:]]
                    # line_num += 1
            # sum_list /= line_num
            hlf_dic.setdefault(int(video.split('_')[0]),sum_list)
    with open('HLF346_ExpandedData.pickle','wb') as f:
        pickle.dump(hlf_dic, f, pickle.HIGHEST_PROTOCOL)



def load_related_video(filelist):
    dic = {}
    for file in filelist:
        with open(file, 'r') as f:
            next(f)
            for eachLine in f:
                a = eachLine.strip().strip('  ').split('  ')
                if len(a) >= 2:
                    dic[int(a[0])] = map(int, a[1].split(' '))
    with open('relatedvideo.pickle', 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)



# filelist = ['Textual_Features/200812/relatedvideo_200812.txt',
#             'Textual_Features/200901/relatedvideo_200901.txt',
#             'Textual_Features/200902/relatedvideo_200902.txt']
# load_related_video(filelist)

def load_statistic(filelist):
    label2videolist = {}
    video2label = {}
    path2videolist = {}
    for file in filelist:
        with open(file, 'r') as f:
            next(f)
            for eachLine in f:
                a = re.split(' |\t',eachLine.strip())
                assert len(a) == 9
                label2videolist.setdefault(a[6], []).append(int(a[0]))
                video2label[int(a[0])] = a[6]
                path2videolist.setdefault(a[8], []).append(int(a[0]))
    with open('label2videolist.pickle', 'wb') as f:
        pickle.dump(label2videolist, f, pickle.HIGHEST_PROTOCOL)
    with open('video2label.pickle', 'wb') as f:
        pickle.dump(video2label, f, pickle.HIGHEST_PROTOCOL)
    with open('path2videolist.pickle', 'wb') as f:
        pickle.dump(path2videolist, f, pickle.HIGHEST_PROTOCOL)

# filelist = ['Textual_Features/200812/statistic_200812_core.txt',
#             'Textual_Features/200901/statistic_200901_core.txt',
#             'Textual_Features/200902/statistic_200902_core.txt',
#             'Textual_Features/200812/statistic_200812_expanded.txt',
#             'Textual_Features/200901/statistic_200901_expanded.txt',
#             'Textual_Features/200902/statistic_200902_expanded.txt',
#             'Textual_Features/200812/statistic_200812_expanded_sa.txt',
#             'Textual_Features/200901/statistic_200901_expanded_sa.txt',
#             'Textual_Features/200902/statistic_200902_expanded_sa.txt',
#             ]
# load_statistic(filelist)




'''
# generate video set
video_list = []
with open('path2videolist.pickle', 'rb') as f:
    data = pickle.load(f)
    del(data['none'])
    for path in data.keys():
        video_list.extend(data[path])
video_set = set(video_list)
print(len(video_set))
with open('videolist.pickle', 'wb') as f:
    pickle.dump(video_set, f, pickle.HIGHEST_PROTOCOL)
'''

# HLF expanded
def select_hlf(filelist):
    hlf_dic = {}
    with open('videolist.pickle', 'rb') as f:
        videoset = pickle.load(f)
    for file in filelist:
        with open(file, 'rb') as f:
            hlf_data = pickle.load(f)
        for i in videoset:
            if i in hlf_data:
                hlf_dic.setdefault(i, hlf_data[i])
    with open('hlf.pickle', 'wb') as f:
        pickle.dump(hlf_dic, f, pickle.HIGHEST_PROTOCOL)

# filelist = ['HLF346_CoreData.pickle', 'HLF346_ExpandedData.pickle']
# select_hlf(filelist)

# with open('hlf.pickle','rb') as f:
#     data = pickle.load(f)
#     print(sum(data[3106808]))
#     print(sum(data[3147780]))

# with open('label2videolist.pickle', 'rb') as f:
#     data = pickle.load(f)
#     print(data.keys())

label_list =['Gaming', 'Education', 'Pets&Animals',
             'News&Politics', 'Entertainment', 'Science&Technology',
             'Travel&Events', 'Sports', 'Trailers',
             'Movies', 'Autos&Vehicles', 'Howto&Style',
             'Music', 'People&Blogs', 'Nonprofits&Activism',
             'Comedy', 'Film&Animation', 'Shows']

'''
with open('video2label.pickle', 'rb') as f:
    data = pickle.load(f)
with open('video_with_vis&txt.pickle','rb') as f:
    video_set = set(pickle.load(f))
video2label = {}
label=[]
for i in video_set:
    video2label.setdefault(i,data[i])
    label.append(data[i])
with open('video2label_vis&txt.pickle','wb') as f:
    pickle.dump(video2label, f, pickle.HIGHEST_PROTOCOL)
print(collections.Counter(label))
'''

filelist = ['Textual_Features/200812/statistic_200812_core.txt',
            'Textual_Features/200901/statistic_200901_core.txt',
            'Textual_Features/200902/statistic_200902_core.txt',
            'Textual_Features/200812/statistic_200812_expanded.txt',
            'Textual_Features/200901/statistic_200901_expanded.txt',
            'Textual_Features/200902/statistic_200902_expanded.txt',
            'Textual_Features/200812/statistic_200812_expanded_sa.txt',
            'Textual_Features/200901/statistic_200901_expanded_sa.txt',
            'Textual_Features/200902/statistic_200902_expanded_sa.txt',
            ]


# generate video id dictionary
def generate_vid_dict(filelist):
    vid_dict = {}
    vid = 0
    vid_set = set()
    with open('video_with_vis&txt.pickle', 'rb') as f:
        video_set = set(pickle.load(f))
    print(len(video_set))
    for filename in filelist:
        with open(filename, 'r') as f:
            next(f)
            for eachLine in f:
                a = re.split(' |\t', eachLine.strip())
                assert len(a) == 9
                if int(a[0]) not in video_set or int(a[0]) in vid_set:
                    continue
                else:
                    vid_dict[int(a[0])] = vid
                    vid_set.add(int(a[0]))
                    vid += 1
    print(len(vid_dict))
    print(vid)
    assert len(vid_dict) == 44144
    return vid_dict


# generate author id dictionary
def generate_aid_dict():
    aid_dict = {}
    aid = 44144
    aid_set = set()
    with open('author.pickle', 'rb') as f:
        author_set = set(pickle.load(f))
    print(len(author_set))
    author_list = list(author_set)
    for a in author_list:
        if a in aid_set:
            continue
        else:
            aid_dict[a] = aid
            aid_set.add(a)
            aid += 1
    print (len(aid_dict))
    print(aid)
    assert len(aid_dict) == 20195
    return aid_dict


# generate label id dictionary
def generate_lid_dict():
    lid_dict = {}
    lid_set=set()
    lid = 64339
    label_list = ['Gaming', 'Education', 'Pets&Animals',
                  'News&Politics', 'Entertainment', 'Science&Technology',
                  'Travel&Events', 'Sports', 'Trailers',
                  'Movies', 'Autos&Vehicles', 'Howto&Style',
                  'Music', 'People&Blogs', 'Nonprofits&Activism',
                  'Comedy', 'Film&Animation', 'Shows']
    for l in label_list:
        lid_dict[l] = lid
        lid +=1
    assert len(lid_dict) == 18
    print(lid)
    return lid_dict

# lid_dict = generate_lid_dict()
# print(lid_dict)
# with open('lid_dict.pickle', 'wb') as f:
#     pickle.dump(lid_dict, f, pickle.HIGHEST_PROTOCOL)
#
# aid_dict = generate_aid_dict()
# print aid_dict['VyrusTv']
# with open('aid_dict.pickle', 'wb') as f:
#     pickle.dump(aid_dict, f, pickle.HIGHEST_PROTOCOL)



# vid_dict = generate_vid_dict(filelist)
# with open('vid_dict.pickle','wb') as f:
#     pickle.dump(vid_dict, f, pickle.HIGHEST_PROTOCOL)


# hlf_dict = {}
# with open('hlf.pickle', 'rb') as f:
#     hlf_data = pickle.load(f)
# with open('vid_dict.pickle','rb') as f:
#     vid_dict = pickle.load(f)
# for vid in vid_dict:
#     hlf_dict.setdefault(vid_dict[vid],hlf_data[vid])
# assert len(hlf_dict) == 44144
# with open('hlf_vis&txt.pickle', 'wb') as f:
#     pickle.dump(hlf_dict, f, pickle.HIGHEST_PROTOCOL)


# with open('hlf_vis&txt.pickle','rb') as f:
#     data = pickle.load(f)
#     print(len(data))
#     print(data.keys()[-1])

# generate context
def generate_context():
    context_dict = {}
    with open('video_with_vis&txt.pickle', 'rb') as f:
        video_list = pickle.load(f)
    with open('video2label_vis&txt.pickle', 'rb') as f:
        v2l_dict = pickle.load(f)
    with open('relate_pair.pickle', 'rb') as f:
        relate_set = pickle.load(f)
    with open('up_pair.pickle', 'rb') as f:
        va_set = pickle.load(f)
    with open('vid_dict.pickle','rb') as f:
        vid_dict = pickle.load(f)
    with open('aid_dict.pickle','rb') as f:
        aid_dict = pickle.load(f)
    with open('lid_dict.pickle', 'rb') as f:
        lid_dict = pickle.load(f)
    v2a_dict = {}
    for pair in list(va_set):
        v2a_dict.setdefault(pair[0], aid_dict[pair[1]])
    v2v_dict = {}
    for pair in list(relate_set):
        v2v_dict.setdefault(pair[0], []).append(vid_dict[pair[1]])
    for vid in video_list:
        vdict = {}
        vdict.setdefault(0, lid_dict[v2l_dict[vid]])
        vdict.setdefault(1, v2v_dict[vid] if vid in v2v_dict else [])
        vdict.setdefault(2, v2a_dict[vid])
        context_dict.setdefault(vid_dict[vid], vdict)
    print(len(context_dict))
    assert len(context_dict) == 44144
    return context_dict

# context_dict = generate_context()
# with open('context_dict.pickle', 'wb') as f:
#     pickle.dump(context_dict, f, pickle.HIGHEST_PROTOCOL)

# with open('context_dict.pickle' ,'rb') as f:
#     data = pickle.load(f)
#     for i in range(10):
#         print data[i]