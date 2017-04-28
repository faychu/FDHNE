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

relate_files = ['Textual_Features/200812/relatedvideo_200812.txt',
                'Textual_Features/200901/relatedvideo_200901.txt',
                'Textual_Features/200902/relatedvideo_200902.txt']
def generate_relate(filelist):
    relate_set = set()
    video_rel = set()
    with open('videolist.pickle', 'rb') as f:
        videolist = pickle.load(f)
    with open('video_with_vis&txt.pickle', 'rb') as f:
        video_vistxt = pickle.load(f)
    for file in filelist:
        with open(file, 'r') as f:
            next(f)
            for eachLine in f:
                v = eachLine.strip().split('  ')
                # assert len(v) ==2
                if len(v) == 2:
                    vlist = v[1].split(' ')
                    for i in vlist:
                        if int(v[0]) in video_vistxt and int(i) in videolist:
                            relate_set.add((int(v[0]), int(i)))
                            relate_set.add((int(i), int(v[0])))
                            video_rel.add(int(i))
                            video_rel.add(int(v[0]))
    with open('relate_pair_all.pickle', 'wb') as f:
        pickle.dump(relate_set, f, pickle.HIGHEST_PROTOCOL)
    with open('video_rel_all.pickle', 'wb') as f:
        pickle.dump(video_rel, f, pickle.HIGHEST_PROTOCOL)
    print('video with relate: %d ' % len(video_rel))
    print('related pair num: %d ' % len(relate_set))

# generate_relate(relate_files)


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
def generate_up(statistic_files):
    author_set = set()
    up_pair_set = set()
    with open('videolist.pickle', 'rb') as f:
        videolist = pickle.load(f)
    with open('video_with_vis&txt.pickle', 'rb') as f:
        video_vistxt = pickle.load(f)
    for file in statistic_files:
        with open(file, 'r') as f:
            next(f)
            for eachLine in f:
                a = re.split(' |\t', eachLine.strip())
                if int(a[0]) in videolist:
                    up_pair_set.add((int(a[0]), a[2]))
                    author_set.add(a[2])
    with open('up_pair_all.pickle', 'wb') as f:
        pickle.dump(up_pair_set, f, pickle.HIGHEST_PROTOCOL)
    with open('author_all.pickle', 'wb') as f:
        pickle.dump(author_set, f, pickle.HIGHEST_PROTOCOL)
    print('up pair: %d ' % len(up_pair_set))
    print('author num: %d ' % len(author_set))

# generate_up(filelist)
topic_set = []
count = 0
with open('topic_vid-for_release.txt', 'r') as f:
    for eachLine in f:
        topic_set.extend([i for i in eachLine.strip().strip('\t').split('\t')[1:-1]])
print(len(topic_set))
with open('video_with_vis&txt.pickle', 'rb') as f:
    video_vistxt = pickle.load(f)
with open('videolist.pickle', 'rb') as f:
    videolist = pickle.load(f)
for i in topic_set:
    if int(i) not in videolist:
        count += 1
print(count)