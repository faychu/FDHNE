videolist.pickle: 类型set，共74088条，即有视觉的video id
video_with_vis&txt.pickle：类型list，共44144条，即具有视觉和文本特征的video id
relate_pair.pickle：相关关系对，类型set，例如((1,2),...)，里面的视频id均有视觉和文本特征，共84214条，正反均有
video_rel.pickle：具有relate关系的视频（且均有视觉和文本特征），共31362条
relate_set_all.pickle：相关关系对，类型set，例如((1,2),...)，里面的视频id均有视觉和文本特征，共116960条，正反均有
video_rel_all.pickle：具有relate关系的视频（且均有视觉和文本特征），共43864条
up_pair_all.pickle：类型set，（vid, authorid），针对具有视觉特征的video，共74088条
author_all.pickle：类型set，针对具有视觉特征的video，共34460个author

up_pair.pickle：类型set，（vid, authorid），针对具有视觉及文本特征的video，共44144条
author.pickle：类型set，针对具有视觉及文本特征的video，共20195个author

video2label_vis&txt.pickle:类型dict，{原始vid：原始label}


标号字典：
vid_dict.pickle：类型dict，（原始id：新id（0-44143））
aid_dict.pickle：类型dict，（原始id：新id（44144-64338））
lid_dict.pickle：类型dict，（原始id：新id（64339-64356））


特征：
hlf_vis&txt.pickle：类型dict，针对具有视觉及文本特征的video的hlf特征，key值范围（0-44143）
tfidf.pickle：类型dict，针对针对具有视觉及文本特征的video的tfidf特征，key值范围（0-44143）

Context：
context.pickle：类型dict，{vid：{0:lid,1:[vid],2:aid}}

输入模型中只需要：
hlf_vis&txt.pickle
tfidf.pickle
context.pickle

类别信息：
Counter({
'Entertainment': 7462,
'News&Politics': 4219,
'Comedy': 4122,
'Music': 3988,
'People&Blogs': 3107,
'Howto&Style': 3076,
'Gaming': 2893,
'Sports': 2710,
'Science&Technology': 2449,
'Autos&Vehicles': 2398,
'Pets&Animals': 1918,
'Film&Animation': 1911,
'Education': 1439,
'Nonprofits&Activism': 1183,
'Travel&Events': 1164,
'Shows': 93,
'Trailers': 7,
'Movies': 5})
