# coding:utf-8
from six.moves import cPickle as pickle
import collections
import numpy as np
import random
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # 实现Matplotlib绘图并保存图像但不显示图形的方法
import scipy as sk

# step 1: load data
context_size = 64357
batch_size = 128
k = 5  # negative samples
with open('hlf_vis&txt.pickle', 'rb') as f:
    hlf_dic = pickle.load(f)
with open('tfidf.pickle', 'rb') as f:
    tfidf_dic = pickle.load(f)
with open('context_dict.pickle', 'rb') as f:
    context_dic = pickle.load(f)

# create a partition vector, train-0, test-1
test_size = 17658
partitions = [0] * 44144
partitions[:test_size] = [1] * test_size
random.shuffle(partitions)


def one_hot(label):
    label_vector = np.zeros([18], dtype=np.float32)
    label_vector[label-64339] = 1
    label_vector.astype(np.float32)
    return label_vector


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord():
    writer0 = tf.python_io.TFRecordWriter('fdhne_train.tfrecord')
    writer1 = tf.python_io.TFRecordWriter('fdhne_test.tfrecord')
    for vid in range(44144):
        vid = np.int64(vid)
        label = context_dic[vid][0]  # from 64339 to 64356
        label_vec = one_hot(label)
        label_vec.astype(np.float32)
        label_vec = label_vec.tostring()
        hlf = hlf_dic[vid]
        hlf = np.array(hlf, dtype=np.float32)
        hlf_raw = hlf.tostring()
        tfidf = tfidf_dic[vid]
        tfidf = tfidf.astype(np.float32)
        tfidf_raw = tfidf.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'vid': _int64_feature(vid),
                         'lvec': _bytes_feature(label_vec),
                         'hlf': _bytes_feature(hlf_raw),
                         'tfidf': _bytes_feature(tfidf_raw)}
            )
        )
        if partitions[vid] == 0:
            writer0.write(example.SerializeToString())
        if partitions[vid] == 1:
            writer1.write(example.SerializeToString())
    writer0.close()
    writer1.close()

# convert_to_tfrecord()

# step 2: function to generate a training batch
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


video_context = set(range(0, 44144))  # video context set (0-44143)
label_context = set(range(64339, 64357))  # label context set (64339-64356)
author_context = set(range(44144, 64339))  # author context set (44144-64338)


def generate_context(vid, r=0, r1=0.3, r2=0.6, k=5):
    a = context_dic[vid]
    r = random.uniform(0, 1) if r == 0 else r
    negative_list = []  # [context]
    if r <= r1:  # label
        context = a[0]
        for i in range(k):  # negative sample
            negative_list.append(random.choice(list(label_context-set([a[0]]))))
    elif r1 < r <= r1+r2:  # video
        if len(a[1]) == 0:
            context = 64357
            for i in range(k):
                negative_list.append(context)
            assert len(negative_list) == k
            return r, context, negative_list
        else:
            context = random.choice(a[1])
            for i in range(k): # negative sample
                negative_list.append(random.choice(list(label_context - set(a[1]))))
    else:  # author
        context = a[2]
        for i in range(k):  # negative sample
            negative_list.append(random.choice(list(label_context - set([a[2]]))))
    assert len(negative_list) == k
    return 0, context, negative_list

# step 3: Build and train FDHNE model

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)  # 变量的初始值为截断正太分布
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Define the placeholders
Xv = tf.placeholder(tf.float32, [None, 346], name='x_v')
Xt = tf.placeholder(tf.float32, [None, 2000], name='x_t')
y_ = tf.placeholder(tf.float32, [None, 18], name='y_')
video_id = tf.placeholder(tf.int64, [None,1], name='vid')
context_inputs = tf.placeholder(tf.int32, shape=[None], name='context_inputs')  # 只是下标
negative_inputs = tf.placeholder(tf.int32, shape=[None, k], name='negative_inputs')
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Define the weights for the layers, and construct the layers with ELU activations.
"""
# 纯视觉
"""
# 纯视觉第一层，输入维数346，输出维数265
W_vl1 = weight_variable([346, 265])
b_vl1 = bias_variable([265])
h_vl1 = tf.nn.elu(tf.matmul(Xv, W_vl1)+b_vl1)
# 纯视觉第二层，输入维数265，输出维数180
W_vl2 = weight_variable([265, 180])
b_vl2 = bias_variable([180])
h_vl2 = tf.nn.elu(tf.matmul(h_vl1, W_vl2)+b_vl2)
# 纯视觉第三层，输入维数180，输出维数100
W_vl3 = weight_variable([180, 100])
b_vl3 = bias_variable([100])
h_vl3 = tf.nn.elu(tf.matmul(h_vl2, W_vl3)+b_vl3)
# 纯视觉第四层，输入维数100，输出维数20
W_vl4 = weight_variable([100, 20])
b_vl4 = bias_variable([20])
h_vl4 = tf.nn.elu(tf.matmul(h_vl3, W_vl4)+b_vl4)
"""
# 视觉嵌入
"""
# 视觉嵌入第一层，输入维数346，输出维数265
W_ml1 = weight_variable([346, 265])
b_ml1 = bias_variable([265])
h_ml1 = tf.nn.elu(tf.matmul(Xv, W_ml1)+b_ml1)
# 视觉嵌入第二层，输入维数265，输出维数180
W_ml2 = weight_variable([265, 180])
b_ml2 = bias_variable([180])
h_ml2 = tf.nn.elu(tf.matmul(h_ml1, W_ml2)+b_ml2)
# 视觉嵌入第三层，输入维数180，输出维数100
W_ml3 = weight_variable([180, 100])
b_ml3 = bias_variable([100])
h_ml3 = tf.nn.elu(tf.matmul(h_ml2, W_ml3)+b_ml3)
# 视觉嵌入第四层，输入维数100，输出维数20
W_ml4 = weight_variable([100, 20])
b_ml4 = bias_variable([20])
h_ml4 = tf.nn.elu(tf.matmul(h_ml3, W_ml4)+b_ml4)

"""
# 文本嵌入
注意：输出维数要与视觉嵌入输出维数一致
"""
# 文本嵌入第一层，输入维数2000，输出维数1500
W_tl1 = weight_variable([2000, 1500])
b_tl1 = bias_variable([1500])
h_tl1 = tf.nn.elu(tf.matmul(Xt, W_tl1)+b_tl1)
# 文本嵌入第二层，输入维数1500，输出维数1000
W_tl2 = weight_variable([1500, 1000])
b_tl2 = bias_variable([1000])
h_tl2 = tf.nn.elu(tf.matmul(h_tl1, W_tl2)+b_tl2)
# 文本嵌入第三层，输入维数1000，输出维数180
W_tl3 = weight_variable([1000, 180])
b_tl3 = bias_variable([180])
h_tl3 = tf.nn.elu(tf.matmul(h_tl2, W_tl3)+b_tl3)
"""
# 联合嵌入
"""
# softmax层，输入维数20+20 = 40，输出维数为18
h_c = tf.concat([h_vl4, h_ml4], 1)
h_c_drop = tf.nn.dropout(h_tl3, keep_prob)
W = weight_variable([346, 18])
b = bias_variable([18])
# y = tf.matmul(h_c_drop, W) + b
y = tf.nn.softmax(tf.matmul(Xv, W) + b)  # 使用softmax作为多分类激活函数，在Ls中体现


def get_context_embedding(context_inputs, negative_inputs):
    embeddings = tf.Variable(tf.random_uniform([context_size, 180], -1.0, 1.0))
    zero_embedding = tf.zeros([1,180])
    embeddings = tf.concat([embeddings, zero_embedding],0)
    W_c = tf.nn.embedding_lookup(embeddings, context_inputs)
    W_n = tf.nn.embedding_lookup(embeddings, negative_inputs)
    tf.reshape(W_n, shape=[batch_size*k, 180])
    return W_c, W_n


# loss function# todo!
# L1st_loss = -tf.reduce_mean(tf.log(tf.nn.softmax(tf.reduce_mean(h_tl3*h_ml2, axis=0))))/batch_size
# W_c, W_n = get_context_embedding(context_inputs, negative_inputs)
# h_ml2_repeat = h_ml2
# L2nd_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(tf.reduce_mean(W_c * h_ml2, axis=0))))  # todo!
# Ls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
Ls = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-10), reduction_indices=[1]))
# Ls = -tf.reduce_sum(y_*y)
joint_loss = Ls
# optimizers
Optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(joint_loss)
# accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# embedding
#  todo!

# Joint Training
filename_queue = tf.train.string_input_producer(['fdhne_train.tfrecord'], num_epochs=None)
vid, lvec, hlf, tfidf = read_and_decode(filename_queue)
_, test_y, test_xv, _ = read_and_decode(tf.train.string_input_producer(['fdhne_test.tfrecord'], num_epochs=None))
vid_var, lvec_var, hlf_var, tfidf_var = tf.train.batch([vid, lvec, hlf, tfidf],
                                       batch_size=batch_size,
                                       num_threads=2,
                                       capacity=1000 + 3 * batch_size)
num_steps = 100000
with tf.Session() as session:
    session.run(tf.global_variables_initializer())  # initialize
    print('Initialized')
    average_loss = 0
    # coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session)
    # threads = tf.train.start_queue_runners(sess=session, coord=coord)
    for step in xrange(num_steps):
        vid, batch_labels, batch_xv, batch_xt = session.run([vid_var, lvec_var, hlf_var, tfidf_var])
        # print (vid,batch_labels,batch_xv,batch_xt)
        r = 0
        contexts = []
        negatives = []
        for i in range(batch_size):
            r, context, negative_list = generate_context(vid[i], r)  # todo!
            contexts.append(context)
            negatives.append(negative_list)
        assert len(contexts) == batch_size
        contexts = np.array(contexts)
        contexts.astype(dtype=np.int32)
        negatives = np.array(negatives)
        negatives.astype(dtype=np.int32)
        feed_dict = {
            Xv: batch_xv,
            Xt: batch_xt,
            y_: batch_labels,
            context_inputs: contexts,  # shape=[batch_size]
            negative_inputs: negatives,  # shape=[batch_size, k]
            keep_prob: 1.0
        }
        _, loss_val, train_accuracy = session.run([Optimizer, joint_loss, accuracy], feed_dict=feed_dict)
        # print loss_val
        average_loss += loss_val

        if step % 100 == 0:
            if step > 0:
                average_loss/=100
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ": ", average_loss)
            print("step %d, training accuracy %g" % (step, train_accuracy))
            average_loss = 0

    # coord.request_stop()
    # coord.join(threads)
    #         _, test_y, test_xv, _ = read_and_decode(tf.train.string_input_producer(['fdhne_test.tfrecord'], num_epochs=None))
    #         tf.train.start_queue_runners(sess=session)
    #         test_y, test_xv = session.run([test_y, test_xv])
    #         print("test accuracy %g" % accuracy.eval(feed_dict={
    #             Xv: test_xv, y_:test_y, keep_prob: 1.0}))
