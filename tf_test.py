# __author__ = 'fay'
import tensorflow as tf
import numpy as np
import random
from six.moves import cPickle as pickle

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'vid': tf.FixedLenFeature([], tf.int64),
                                           'lvec': tf.FixedLenFeature([], tf.string),
                                           'hlf': tf.FixedLenFeature([], tf.string),
                                           'tfidf': tf.FixedLenFeature([], tf.string)
                                       })
    hlf = tf.decode_raw(features['hlf'], tf.float32)
    hlf.set_shape([346])
    lvec = tf.decode_raw(features['lvec'], tf.float32)
    lvec.set_shape([18])
    vid = features['vid']
    tfidf = tf.decode_raw(features['tfidf'], tf.float32)
    tfidf.set_shape([2000])
    return vid, lvec, hlf, tfidf

def generate_batch(batch_size, shuffle=False):  # todo!
    filename_queue = tf.train.string_input_producer(['fdhne_train.tfrecord'], num_epochs=None)
    vid, lvec, hlf, tfidf = read_and_decode(filename_queue)
    if shuffle:
        vid, lvec, hlf, tfidf = tf.train.shuffle_batch([vid, lvec, hlf, tfidf],
                                                       batch_size=batch_size,
                                                       num_threads=2,
                                                       capacity=1000+3*batch_size)
    else:
        vid, lvec, hlf, tfidf = tf.train.batch([vid, lvec, hlf, tfidf],
                                               batch_size=batch_size,
                                               num_threads=2,
                                               capacity=1000+3*batch_size)
    return vid, lvec, hlf, tfidf
#
# writer = tf.python_io.TFRecordWriter('test.tfrecord')
# with open('sample', 'r') as f:
#     for eachLine in f:
#         data = eachLine.strip().split('\t')
#         id = int(data[0])
#         print(id)
#         label = data[-1].encode()
#         feat = np.random.randn(10)
#         feat = feat.astype(np.float)
#         feat = feat.tostring()
#         print(type(feat))
#         rel = list(range(random.randint(0, 5)))
#         rel = str(rel).encode()
#         print('id: ', id, ',label:', label, 'feat:', feat)
#         example = tf.train.Example(
#             features=tf.train.Features(
#                 feature={'id': _int64_feature(id),
#                          'label': _bytes_feature(label),
#                          'feature': _bytes_feature(feat),
#                          'rel': _bytes_feature(rel)}
#             )
#         )
#         writer.write(example.SerializeToString())
#     writer.close()
#
#
# def read_and_decode(filename_queue):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'id': tf.FixedLenFeature([], tf.int64),
#                                            'label': tf.FixedLenFeature([], tf.string),
#                                            'feature': tf.FixedLenFeature([], tf.string),
#                                            'rel': tf.FixedLenFeature([], tf.string)
#                                        })
#     feat = tf.decode_raw(features['feature'], tf.float64)
#     feat.set_shape([10])
#     print('hlf',feat[0])
#     label = features['label']
#     id = features['id']
#     print('vid',id)
#     rel = features['rel']
#     return id, label, feat, rel
#
#
# filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
# id, label, feat, rel = read_and_decode(filename_queue)
# id, label, feat, rel = tf.train.batch([id, label, feat, rel], batch_size=5, num_threads=2, capacity=1000)
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# tf.train.start_queue_runners(sess=sess)
# for i in range(4):
#     id_val, label_val, feat_val, rel_val = sess.run([id, label, feat, rel])
#     print('id: ', id_val, ',label:', label_val[0], 'feat:', feat_val[0], ',rel:', type(rel_val[0]))
#
#
# # # test_set_size = 4
# # # # create a partition vector
# # # partitions = [0] * 14
# # # partitions[:test_set_size]=[1]*test_set_size
# # # random.shuffle(partitions)
# #
# # def one_hot(label):
# #     label_vector = np.zeros([18], dtype=np.float64)
# #     label_vector[label-64339] = 1
# #     return label_vector
# #
# # a = one_hot(64356)
# # print a
#
# embedding = tf.constant(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))  # shape(3,4)
# ids = tf.constant([[0,1],[2,0],[0,2]])
# with tf.Session():
#     # a = tf.nn.embedding_lookup(embedding, ids)
#     # print a.eval()
#     b=[]
#     for i in range(3):
#         b.append(tf.reduce_sum(tf.nn.embedding_lookup(embedding, ids[i]),axis=0))
#     print tf.stack(b).eval()



with tf.Session() as session:
    session.run(tf.global_variables_initializer())  # initialize
    vid, batch_labels, batch_xv, batch_xt = generate_batch(5)
    tf.train.start_queue_runners(sess=session)
    vid, batch_labels, batch_xv, batch_xt=session.run([vid, batch_labels, batch_xv, batch_xt])
    print vid