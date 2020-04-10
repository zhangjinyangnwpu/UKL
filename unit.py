import numpy as np
import tensorflow as tf

def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def af(x):
    return tf.nn.swish(x)

def classifer_share3d_3(feature,training,reuse=True,cube_size=3):
        print(feature)
        feature = tf.expand_dims(feature, 4)
        f_num = 64
        with tf.variable_scope('classifer', reuse=reuse):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(feature, f_num, (cube_size,1,8), strides=(1,1,3), padding='valid')
                conv0 = tf.layers.batch_normalization(conv0,training=training)
                conv0 = af(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0, f_num * 2, (1,cube_size,3), strides=(1,1,2), padding='valid')
                conv1 = tf.layers.batch_normalization(conv1,training=training)
                conv1 = af(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(conv1, f_num * 4, (1,1,3), strides=(1,1,2), padding='valid')
                conv2 = tf.layers.batch_normalization(conv2,training=training)
                conv2 = af(conv2)
                print(conv2)
            with tf.variable_scope('global_info'):
                f_shape = int(conv2.get_shape().as_list()[3])
                feature = tf.layers.conv3d(conv2, f_num * 8, (1,1,f_shape), (1,1,1))
                feature = tf.layers.flatten(feature)
                print(feature)
        return feature


def classifer_share3d_6(feature, training, reuse=True, cube_size=3):
    print(feature)
    feature = tf.expand_dims(feature, 4)
    f_num = 64
    with tf.variable_scope('classifer', reuse=reuse):
        with tf.variable_scope('conv00'):
            conv0 = tf.layers.conv3d(feature, f_num, (cube_size, 1, 8), strides=(1, 1, 3), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv01'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv10'):
            conv1 = tf.layers.conv3d(conv0, f_num * 2, (1, cube_size, 3), strides=(1, 1, 2), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv11'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv20'):
            conv2 = tf.layers.conv3d(conv1, f_num * 4, (1, 1, 3), strides=(1, 1, 2), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv21'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 2), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('global_info'):
            f_shape = int(conv2.get_shape().as_list()[3])
            feature = tf.layers.conv3d(conv2, f_num * 8, (1, 1, f_shape), (1, 1, 1))
            feature = tf.layers.flatten(feature)
            print(feature)
    return feature

def classifer_share3d_9(feature, training, reuse=True, cube_size=3):
    print(feature)
    feature = tf.expand_dims(feature, 4)
    f_num = 64
    with tf.variable_scope('classifer', reuse=reuse):
        with tf.variable_scope('conv00'):
            conv0 = tf.layers.conv3d(feature, f_num, (cube_size, 1, 8), strides=(1, 1, 3), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv01'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv02'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv10'):
            conv1 = tf.layers.conv3d(conv0, f_num * 2, (1, cube_size, 3), strides=(1, 1, 2), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv11'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv12'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv20'):
            conv2 = tf.layers.conv3d(conv1, f_num * 4, (1, 1, 3), strides=(1, 1, 2), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv21'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv22'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('global_info'):
            f_shape = int(conv2.get_shape().as_list()[3])
            feature = tf.layers.conv3d(conv2, f_num * 8, (1, 1, f_shape), (1, 1, 1))
            feature = tf.layers.flatten(feature)
            print(feature)
    return feature

def classifer_share3d_12(feature, training, reuse=True, cube_size=3):
    print(feature)
    feature = tf.expand_dims(feature, 4)
    f_num = 64
    with tf.variable_scope('classifer', reuse=reuse):
        with tf.variable_scope('conv00'):
            conv0 = tf.layers.conv3d(feature, f_num, (cube_size, 1, 8), strides=(1, 1, 3), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv01'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv02'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv03'):
            conv0 = tf.layers.conv3d(conv0, f_num, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv0 = tf.layers.batch_normalization(conv0, training=training)
            conv0 = af(conv0)
            print(conv0)
        with tf.variable_scope('conv10'):
            conv1 = tf.layers.conv3d(conv0, f_num * 2, (1, cube_size, 3), strides=(1, 1, 2), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv11'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv12'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv13'):
            conv1 = tf.layers.conv3d(conv1, f_num * 2, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = af(conv1)
            print(conv1)
        with tf.variable_scope('conv20'):
            conv2 = tf.layers.conv3d(conv1, f_num * 4, (1, 1, 3), strides=(1, 1, 2), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv21'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='valid')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv22'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('conv23'):
            conv2 = tf.layers.conv3d(conv2, f_num * 4, (1, 1, 3), strides=(1, 1, 1), padding='same')
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = af(conv2)
            print(conv2)
        with tf.variable_scope('global_info'):
            f_shape = int(conv2.get_shape().as_list()[3])
            feature = tf.layers.conv3d(conv2, f_num * 8, (1, 1, f_shape), (1, 1, 1))
            feature = tf.layers.flatten(feature)
            print(feature)
    return feature


def classifer_cluster(feature,training,cluster_num,reuse=False):
    with tf.variable_scope('classifer_cluster', reuse=reuse):
        feature = af(feature)
        with tf.variable_scope('fc'):
            fc = tf.layers.dense(feature,256)
            fc = tf.layers.batch_normalization(fc,training=training)
            fc = af(fc)
        with tf.variable_scope('cluster_pre_label'):
            cluster_pre_label = tf.layers.dense(fc,cluster_num)
    return fc,cluster_pre_label

def classifer_classification(feature,training,class_num,reuse = False):
    with tf.variable_scope('classifer_classification', reuse=reuse):
        feature = af(feature)
        with tf.variable_scope('fc'):
            fc = tf.layers.dense(feature,256)
            fc = tf.layers.batch_normalization(fc,training=training)
            fc = af(fc)
        with tf.variable_scope('classification_pre_label'):
            classification_pre_label = tf.layers.dense(fc,class_num)
    return fc,classification_pre_label

def classifer_combine(fc1,fc2,concate_way,class_num,training,reuse=False):
    with tf.variable_scope('combine', reuse=reuse):
        if concate_way == 0:
            fusion = fc1
        if concate_way == 1:
            fusion = tf.concat([fc1,fc2],axis=1)
        if concate_way == 2:
            fusion = tf.multiply(fc1,fc2)
        if concate_way == 3:
            fc1 = tf.expand_dims(fc1,2)
            fc2 = tf.expand_dims(fc2,2)
            fusion = tf.concat([fc1, fc2], axis=2)
            fusion = tf.expand_dims(fusion,3)
            fusion = tf.layers.conv2d(fusion,1,(1,2),(1,1))
            fusion = tf.layers.flatten(fusion)
        if concate_way == 4:
            fc1 = tf.layers.dense(fc1, 16, activation=af)
            fc1 = tf.layers.batch_normalization(fc1,training=training)
            fc2 = tf.layers.dense(fc2, 32, activation=af)
            fc2 = tf.layers.batch_normalization(fc2,training=training)
            fc1 = tf.expand_dims(fc1,2)
            fc2 = tf.expand_dims(fc2,1)
            fusion = tf.matmul(fc1,fc2)
            fusion = tf.layers.flatten(fusion)
        with tf.variable_scope('logits'):
            logits = tf.layers.dense(fusion,class_num)
    return logits

