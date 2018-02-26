
# coding: utf-8

# In[17]:


import tensorflow as tf
import numpy as np
CKPT_DIR = './vgg16ckpt/'




from tensorflow.python.tools import inspect_checkpoint as chkp


# In[3]:


chkp.print_tensors_in_checkpoint_file("./vgg16ckpt/vgg_16.ckpt", tensor_name='', all_tensors=True)





class vgg16:
    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc8l)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        with tf.name_scope('conv1') as sc:
            # conv1_1
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv1_2
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1
            self.pool1 = tf.nn.max_pool(self.conv1_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool1')


        with tf.name_scope('conv2') as sc:
            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2
            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')


        with tf.name_scope('conv3') as sc:
            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3
            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')
        with tf.name_scope('conv4') as sc:
            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')


        with tf.name_scope('conv5') as sc:
            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool5
            self.pool5 = tf.nn.max_pool(self.conv5_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool5')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc6') as scope:
            # shape = int(np.prod(self.pool5.get_shape()[1:]))
            # fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
            #                                              dtype=tf.float32,
            #                                              stddev=1e-1), name='weights')
            # fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
            #                      trainable=True, name='biases')
            # pool5_flat = tf.reshape(self.pool5, [-1, shape])
            # fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            # self.fc1 = tf.nn.relu(fc1l)
            # self.parameters += [fc1w, fc1b]
            kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.fc6 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # fc2
        with tf.name_scope('fc7') as scope:
            # fc2w = tf.Variable(tf.truncated_normal([1,1,4096, 4096],
            #                                              dtype=tf.float32,
            #                                              stddev=1e-1), name='weights')
            # fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
            #                      trainable=True, name='biases')
            #
            # fc2l = tf.nn.bias_add(tf.matmul(self.fc1, tf.squeeze(fc2w)), fc2b)
            # self.fc2 = tf.nn.relu(fc2l)
            # self.parameters += [fc2w, fc2b]
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.fc6, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.fc7 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # fc3
        with tf.name_scope('fc8') as scope:
            # fc3w = tf.Variable(tf.truncated_normal([1,1,4096, 1000],
            #                                              dtype=tf.float32,
            #                                              stddev=1e-1), name='weights')
            # fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
            #                      trainable=True, name='biases')
            # self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, tf.squeeze(fc3w)), fc3b)
            # self.parameters += [fc3w, fc3b]
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 1000], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.fc7, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.fc8l = out
            self.fc8 = tf.nn.softmax(out, name=scope)
            self.parameters += [kernel, biases]


# In[88]:
input_img = tf.placeholder(tf.float32,[4,224,224,3])

with tf.variable_scope('vgg_16') as scope:
    VGGModel = vgg16(imgs=input_img)



with tf.Session() as sess:
    vars_vgg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')
    saver_vgg = tf.train.Saver(max_to_keep=3, var_list=vars_vgg)
    ckpt_vgg = tf.train.get_checkpoint_state(CKPT_DIR)
    start_itr = 0
    # if ckpt_vgg and ckpt_vgg.model_checkpoint_path:
    saver_vgg.restore(sess, './vgg16ckpt/vgg_16.ckpt')
    print("Model restored...")

    vggfc6feature = VGGModel.fc6


    #then sess run