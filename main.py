
import tensorflow as tf
import csv
import os
import numpy as np
import datetime
from tensorflow.contrib import rnn
from tools.load_data_new import load_content, load_tl, load_img
from sklearn import metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import random
from tools.flip_gradient import flip_gradient
from tensorflow.contrib import layers

np.set_printoptions(threshold=np.inf)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


def load_user(path_dir):
    name_test_tw = []
    name_test_fq = []

    user_path = path_dir + 'train_name_shuffle.csv' #857[tw, fq1, fq2]
    label_path = path_dir + 'train_label_shuffle.csv' #[tw==fq, label=1]
    label_path_re = path_dir + 'train_label_re_shuffle.csv'

    label_path_tw_single = path_dir + 'train_label_tw_d.csv' 
    label_path_fq_single = path_dir + 'train_label_fq_d.csv' 

    label_path_tw_single_test = path_dir + 'test_label_tw_d.csv'  
    label_path_fq_single_test = path_dir + 'test_label_fq_d.csv'

    user_test_path = path_dir + 'test_name_shuffle.csv'
    label_test_path = path_dir + 'test_label_shuffle.csv'
    label_test_path_re = path_dir + 'test_label_re_shuffle.csv'

    # user
    f_user = open(user_path, 'r', encoding='utf-8')
    reader_user = csv.reader(f_user)
    user_lines = [row for row in reader_user]
    loss_dict = ['alexjamesfitz', 'iamedia007', 'elliotj651', 'whizhouse', 'niketa', 'rickyrobinett', 'scottdreyfus',
                 'joekanakaraj', 'theory__', 'persiancowboy', 'libbydoodle', 'genepark', 'johnedetroit', 'djidlemind',
                 'gyrus', 'anytimefitness', 'afpa1', 'dberkowitz', 'iprincemb', 'cherylrice', 'amalucky', 'opajdara',
                 'andrewphelps', 'edgarprado', 'rushwan', 'razabegg', 'megan_calimbas', 'liscallelura', 'rtsnance',
                 'thomastowell', 'sears', '_dom', 'johngarcia', 'visitcedarhill', 'jtw90210', 'bravotv', 'dales777',
                 'andrewvest', 'pgatour', 'agoldfisher', 'nycgov'
                 ] 
    history = []
    for i in range(0,len(user_lines)):
        if user_lines[i][0] in loss_dict:
            history.append(i)
    print(history)
    user_lines = [user_lines[i] for i in range(0, len(user_lines), 1) if i not in history] #821[tw, fq1, fq2]
    # label
    f_label = open(label_path, 'r', encoding='utf-8')
    reader_label = csv.reader(f_label)
    label_lines = [row for row in reader_label]
    label_lines = [label_lines[i] for i in range(0, len(label_lines), 1) if i not in history] #821
    # label-1
    f_label_re = open(label_path_re, 'r', encoding='utf-8')
    reader_label_re = csv.reader(f_label_re)
    label_lines_re = [row for row in reader_label_re]
    label_lines_re = [label_lines_re[i] for i in range(0, len(label_lines_re), 1) if i not in history] #821

    # label_tw
    f_label_tw_single = open(label_path_tw_single, 'r', encoding='utf-8')
    reader_tw_single = csv.reader(f_label_tw_single)
    label_tw_single = [row for row in reader_tw_single]
    label_tw_single = [label_tw_single[i] for i in range(0, len(label_tw_single), 1) if i not in history] #821
    # label_fq
    f_label_fq_single = open(label_path_fq_single, 'r', encoding='utf-8')
    reader_fq_single = csv.reader(f_label_fq_single)
    label_fq_single = [row for row in reader_fq_single]
    label_fq_single = [label_fq_single[i] for i in range(0, len(label_fq_single), 1) if i not in history] #821


    f_user_test = open(user_test_path, 'r', encoding='utf-8')
    reader_user_test = csv.reader(f_user_test)
    user_lines_test = [row for row in reader_user_test]
    test_loss_dict = ['jorenerene', 'andrewvest', 'rushwan', 'thomastowell', 'johngarcia', 'rtsnance', 'gyrus'] #101[tw, fq1, fq2]
    test_history = []
    for i in range(0, len(user_lines_test)):
        if user_lines_test[i][0] in test_loss_dict:
            test_history.append(i)
    print(test_history)
    user_lines_test = [user_lines_test[i] for i in range(0, len(user_lines_test), 1) if
                            i not in test_history] #101

    # label_tw_test label_path_tw_single_test, label_path_fq_single_test label_tw_single_test, label_fq_single_test
    f_label_tw_single_test = open(label_path_tw_single_test, 'r', encoding='utf-8')
    reader_tw_single_test = csv.reader(f_label_tw_single_test)
    label_tw_single_test = [row for row in reader_tw_single_test]
    label_tw_single_test = [label_tw_single_test[i] for i in range(0, len(label_tw_single_test), 1) if i not in test_history]
    # label_fq_test
    f_label_fq_single_test = open(label_path_fq_single_test, 'r', encoding='utf-8')
    reader_fq_single_test = csv.reader(f_label_fq_single_test)
    label_fq_single_test = [row for row in reader_fq_single_test]
    label_fq_single_test = [label_fq_single_test[i] for i in range(0, len(label_fq_single_test), 1) if
                            i not in test_history]
    # label
    f_label_test = open(label_test_path, 'r', encoding='utf-8')
    reader_label_test = csv.reader(f_label_test)
    label_lines_test = [row for row in reader_label_test]
    label_lines_test = [label_lines_test[i] for i in range(0, len(label_lines_test), 1) if
                            i not in test_history]
    # label-1
    f_label_re_test = open(label_test_path_re, 'r', encoding='utf-8')
    reader_label_re_test = csv.reader(f_label_re_test)
    label_lines_re_test = [row for row in reader_label_re_test]
    label_lines_re_test = [label_lines_re_test[i] for i in range(0, len(label_lines_re_test), 1) if
                            i not in test_history]

    return user_lines, np.float32(label_lines), label_lines_re, user_lines_test, np.float32(label_lines_test), label_lines_re_test, label_tw_single, label_fq_single, label_tw_single_test, label_fq_single_test

def load_user_test(path_dir):
    name_test_tw = []
    name_test_fq = []
    label_path_tw_single_test = path_dir + 'validate_label_tw_d.csv'  
    label_path_fq_single_test = path_dir + 'validate_label_fq_d.csv'

    user_test_path = path_dir + 'validate_name_shuffle.csv'
    label_test_path = path_dir + 'validate_label_shuffle.csv'
    label_test_path_re = path_dir + 'validate_label_re_shuffle.csv'


    f_user_test = open(user_test_path, 'r', encoding='utf-8')
    reader_user_test = csv.reader(f_user_test)
    user_lines_test = [row for row in reader_user_test]
    test_loss_dict = ['visitcedarhill', 'pgatour', 'jtw90210', 'dales777', 'agoldfisher', 'bravotv', 'gyrus']
    f_history = []
    for i in range(0, len(user_lines_test)):
        if user_lines_test[i][0] in test_loss_dict:
            f_history.append(i)
    print(f_history)
    user_lines_test = [user_lines_test[i] for i in range(0, len(user_lines_test), 1) if
                           i not in f_history]

    # label_tw_test
    f_label_tw_single_test = open(label_path_tw_single_test, 'r', encoding='utf-8')
    reader_tw_single_test = csv.reader(f_label_tw_single_test)
    label_tw_single_test = [row for row in reader_tw_single_test]
    label_tw_single_test = [label_tw_single_test[i] for i in range(0, len(label_tw_single_test), 1) if
                       i not in f_history]
    # label_fq_test
    f_label_fq_single_test = open(label_path_fq_single_test, 'r', encoding='utf-8')
    reader_fq_single_test = csv.reader(f_label_fq_single_test)
    label_fq_single_test = [row for row in reader_fq_single_test]
    label_fq_single_test = [label_fq_single_test[i] for i in range(0, len(label_fq_single_test), 1) if
                       i not in f_history]

    # label
    f_label_test = open(label_test_path, 'r', encoding='utf-8')
    reader_label_test = csv.reader(f_label_test)
    label_lines_test = [row for row in reader_label_test]
    label_lines_test = [label_lines_test[i] for i in range(0, len(label_lines_test), 1) if i not in f_history]
    # label-1
    f_label_re_test = open(label_test_path_re, 'r', encoding='utf-8')
    reader_label_re_test = csv.reader(f_label_re_test)
    label_lines_re_test = [row for row in reader_label_re_test]
    label_lines_re_test = [label_lines_re_test[i] for i in range(0, len(label_lines_re_test), 1) if i not in f_history]

    return user_lines_test, np.float32(label_lines_test), label_lines_re_test, label_tw_single_test, label_fq_single_test

def index_dict(index, index_dict_tw, index_dict_fq):
    tw1 = []
    fq1 = []
    fq2 = []
    for pair in index:
        tw1.extend([index_dict_tw[pair[0]]])
        fq1.extend([index_dict_fq[pair[1]]])
        fq2.extend([index_dict_fq[pair[2]]])
    return tw1, fq1, fq2

def  shuffle_data(train_user_, train_label_, label_lines_re_, label_tw_single_train_, label_fq_single_train_):
    state = np.random.get_state()
    np.random.shuffle(train_user_)

    np.random.set_state(state)
    np.random.shuffle(train_label_)
    
    np.random.set_state(state)
    np.random.shuffle(label_lines_re_)
    
    np.random.set_state(state)
    np.random.shuffle(label_tw_single_train_)

    np.random.set_state(state)
    np.random.shuffle(label_fq_single_train_)


    return train_user_, train_label_, label_lines_re_, label_tw_single_train_, label_fq_single_train_


def  shuffle_data_test(test_user_, test_label_, label_lines_re_test_, label_tw_single_test_, label_fq_single_test_):
    state1 = np.random.get_state()
    np.random.shuffle(test_user_)

    np.random.set_state(state1)
    np.random.shuffle(test_label_)
    
    np.random.set_state(state1)
    np.random.shuffle(label_lines_re_test_)
    
    np.random.set_state(state1)
    np.random.shuffle(label_tw_single_test_)

    np.random.set_state(state1)
    np.random.shuffle(label_fq_single_test_)        
    
    return test_user_, test_label_, label_lines_re_test_, label_tw_single_test_, label_fq_single_test_

def  shuffle_data_test2(test_user_, test_label_, label_lines_re_test_, label_tw_single_test_, label_fq_single_test_):
    state1 = np.random.get_state()
    np.random.shuffle(test_user_)

    np.random.set_state(state1)
    np.random.shuffle(test_label_)
    
    np.random.set_state(state1)
    np.random.shuffle(label_lines_re_test_)
    
    np.random.set_state(state1)
    np.random.shuffle(label_tw_single_test_)

    np.random.set_state(state1)
    np.random.shuffle(label_fq_single_test_)        
    
    return test_user_, test_label_, label_lines_re_test_, label_tw_single_test_, label_fq_single_test_


path_dir = '/data/new_dataset2_shuffle/new_dataset2_shuffle/'

path_index_tw = '/data/Network/Network/Origin/user_index_tw.npy'
path_index_fq = '/data/Network/Network/Origin/user_index_fq.npy'

tw_index_lines = np.load(path_index_tw).tolist()
fq_index_lines = np.load(path_index_fq).tolist()


print('------index lines loaded!------')

n_heads = [5, 1] # additional entry for the output layer [8,1]
residual = False
nonlinearity = tf.nn.elu

num_city = 947
num_time = 77

num_epochs = 200 # 200 1000 2000
batch_size = 16 #16 32 64 128

num_filters = 1024 #64--128--256---512---1024
kernel_size = 7 # 2--3---5---7---9
num_tw = 1000
num_fq = 52
num_pair = 4
num_pair_compare = 2
embedding_dim = 768
node_feature = 200
con_rep_di = 200

learning_rate = 0.0001 #0.00001, 0.0001, 0.001, 0.01, 0.1
learning_rate_view = 0.0001


hidden_size = 150
num_index_one_hot = 3
num_test = 153

node_num_tw = 3463
node_num_fq = 3833
img_dim = 2048
bits = 128

max_auc = 0
max_precision = 0
max_recall = 0
max_f1 = 0
max_acc = 0
num_save_epoch = 0
num_save_epoch2 = 0
max_acc_test = 0
max_auc_test = 0
max_precision_test = 0
max_recall_test = 0
max_f1_test = 0
max_acc_test = 0 

path_biases_tw = '/data/Network/Network/Origin/tw_bias_matrix.npy'
path_biases_fq = '/data/Network/Network/Origin/fq_bias_matrix.npy'
biases_tw = np.load(path_biases_tw)
biases_fq = np.load(path_biases_fq)

path_embedding_tw = '/data/Network/Network/Origin/tw_sort.npy'
path_embedding_fq = '/data/Network/Network/Origin/fq_sort.npy'
embedding_tw = np.load(path_embedding_tw)
embedding_fq = np.load(path_embedding_fq)
print('embedding_tw', np.shape(embedding_tw))
print('embedding_fq', np.shape(embedding_fq))

print('------biases adj loaded!------')

weight_tw = np.eye(node_num_tw)[np.newaxis,:,:]
weight_fq = np.eye(node_num_fq)[np.newaxis,:,:]

train_user_, train_label_, label_lines_re_, test_user_, test_label_, label_lines_re_test_, label_tw_single_train_, label_fq_single_train_, label_tw_single_test_, label_fq_single_test_ = load_user(path_dir)
print('------user pair loaded!')
print('train_user_: %s, *****train_label_: %s,*****label_lines_re_: %f*************test_user_: %s------test_label_: %s------label_lines_re_test_: %s------label_tw_single_train_: %s------label_fq_single_train_: %s------label_tw_single_test_: %s------label_fq_single_test_: %s' % (len(train_user_), len(train_label_), len(label_lines_re_), len(test_user_), len(test_label_), len(label_lines_re_test_), len(label_tw_single_train_), len(label_fq_single_train_), len(label_tw_single_test_), len(label_fq_single_test_)))

test_user_2, test_label_2, label_lines_re_test_2, label_tw_single_test_2, label_fq_single_test_2 = load_user_test(path_dir)
print('------test_user pair loaded!')
print('test_user_2: %s, *****test_label_2: %s,*****label_lines_re_test_2: %s*************label_tw_single_test_2 %s------label_fq_single_test_2: %s' % (len(test_user_2), len(test_label_2), len(label_lines_re_test_2), len(label_tw_single_test_2), len(label_fq_single_test_2)))


test_user2, test_label2, label_lines_re_test2, label_tw_single_test2, label_fq_single_test2 = shuffle_data_test2(test_user_2, test_label_2, label_lines_re_test_2, label_tw_single_test_2, label_fq_single_test_2)

lamda_set = [0.1]


for lamda in lamda_set:
    model_path = './model_saved/model_5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/model_5_3_5_5_5_att_2_check_4_6_109_109_2_update_2%s.ckpt'% (str(lamda))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.reset_default_graph()

    train_user, train_label, label_lines_re, label_tw_single_train, label_fq_single_train = shuffle_data(train_user_, train_label_, label_lines_re_, label_tw_single_train_, label_fq_single_train_)

    test_user, test_label, label_lines_re_test, label_tw_single_test, label_fq_single_test = shuffle_data_test(test_user_, test_label_, label_lines_re_test_,  label_tw_single_test_, label_fq_single_test_)

    node_embedding_TW = tf.Variable(tf.constant(0, shape=[node_num_tw, node_feature], dtype=tf.float32), name='gener_node_embedding_tw', trainable=True)
    node_embedding_TW1 = tf.assign(node_embedding_TW, embedding_tw)
    
    node_embedding_FQ = tf.Variable(tf.constant(0, shape=[node_num_fq, node_feature], dtype=tf.float32), name='gener_node_embedding_fq', trainable=True)
    node_embedding_FQ1 = tf.assign(node_embedding_FQ, embedding_fq)



    class MODEL():
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate
            self.learning_rate_view = learning_rate_view

            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.l = tf.placeholder(tf.float32, [])


            self.training = tf.placeholder(dtype=tf.bool, shape=(), name='training')
            self.y_label = tf.placeholder(tf.float32, [None, num_pair_compare], name='y_label')
            self.y_label_pre = tf.placeholder(tf.float32, [None, num_pair_compare], name='y_label_pre')

            self.x_batch_TW1 = tf.placeholder(tf.float32, [None, num_tw, embedding_dim], name='x_batch_TW1')
            self.x_batch_FQ1 = tf.placeholder(tf.float32, [None, num_fq, embedding_dim], name='x_batch_FQ1')
            self.x_batch_FQ2 = tf.placeholder(tf.float32, [None, num_fq, embedding_dim], name='x_batch_FQ2')

            self.x_batch_TW_single = tf.placeholder(tf.int32, [None, 2], name='x_batch_TW_single')  
            self.x_batch_FQ_single = tf.placeholder(tf.int32, [None, 2], name='x_batch_FQ_single')

            self.x_batch_TW1_img = tf.placeholder(tf.float32, [None, img_dim], name='x_batch_TW1_img') 
            self.x_batch_FQ1_img = tf.placeholder(tf.float32, [None, img_dim], name='x_batch_FQ1_img') 
            self.x_batch_FQ2_img = tf.placeholder(tf.float32, [None, img_dim], name='x_batch_FQ2_img')

            self.x_batch_TW1_tl = tf.placeholder(tf.float32, [None, num_time, num_city], name='x_batch_TW1_tl') 
            self.x_batch_FQ1_tl = tf.placeholder(tf.float32, [None, num_time, num_city], name='x_batch_FQ1_tl') 
            self.x_batch_FQ2_tl = tf.placeholder(tf.float32, [None, num_time, num_city], name='x_batch_FQ2_tl') 

            self.index_tw1 = tf.placeholder(tf.int32, [None], name='index_tw1')
            self.index_fq1 = tf.placeholder(tf.int32, [None], name='index_fq1')
            self.index_fq2 = tf.placeholder(tf.int32, [None], name='index_fq2')

            self.cov_pool_TW1 = 25*(self.extract_content(self.x_batch_TW1,  num_filters, kernel_size, name='gener_txt_tw'))
            self.cov_pool_FQ1 = 25*(self.extract_content(self.x_batch_FQ1,  num_filters, kernel_size, name='gener_txt_fq'))
            self.cov_pool_FQ2 = 25*(self.extract_content(self.x_batch_FQ2,  num_filters, kernel_size, name='gener_txt_fq', reuse=True))     

            self.x_batch_TW1_img2 = self.extract_img(self.x_batch_FQ1_img, 1024, name='gener_img_tw')
            self.x_batch_FQ1_img2 = self.extract_img(self.x_batch_FQ1_img, 1024, name='gener_img_fq')
            self.x_batch_FQ2_img2 = self.extract_img(self.x_batch_FQ1_img, 1024, name='gener_img_fq', reuse=True)
            
            self.TW_tl_repres =  250*(self.extract_tl(self.x_batch_TW1_tl, hidden_size, name='gener_tl_tw'))
            self.FQ1_tl_repres = 250*(self.extract_tl(self.x_batch_FQ1_tl, hidden_size, name='gener_tl_fq'))
            self.FQ2_tl_repres = 250*(self.extract_tl(self.x_batch_FQ2_tl, hidden_size, name='gener_tl_fq', reuse=True))

            self.TW_repre_concat = tf.concat([tf.expand_dims(self.cov_pool_TW1, 1), tf.expand_dims(self.x_batch_TW1_img2, 1), tf.expand_dims(self.TW_tl_repres, 1)], 1)

            self.TW_self = tf.ones([self.batch_size, 1])
            self.TW_text_img_sim = tf.expand_dims(self.cosine_distance(self.cov_pool_TW1, self.x_batch_TW1_img2), -1) 
            self.TW_text_tl_sim = tf.expand_dims(self.cosine_distance(self.cov_pool_TW1, self.TW_tl_repres), -1)       
            self.TW_Img_tl_sim = tf.expand_dims(self.cosine_distance(self.x_batch_TW1_img2, self.TW_tl_repres), -1)
            self.TW_A_text = tf.expand_dims(tf.concat([self.TW_self, self.TW_text_img_sim, self.TW_text_tl_sim], -1),1)
            self.TW_A_img = tf.expand_dims(tf.concat([self.TW_text_img_sim, self.TW_self, self.TW_Img_tl_sim], -1),1)
            self.TW_A_tl = tf.expand_dims(tf.concat([self.TW_text_tl_sim, self.TW_Img_tl_sim, self.TW_self], -1),1)
            self.TW_A = tf.concat([self.TW_A_text, self.TW_A_img, self.TW_A_tl], 1)
            
            # TW_A_GCN
            self.TW_gc1_out = self.GraphConvolution(self.TW_repre_concat, 1024, 512, self.TW_A, name='gener_TW_gc1')
            self.TW_gc2_out = self.GraphConvolution(self.TW_gc1_out, 512, 200, self.TW_A, name='gener_TW_gc2')

            # FQ1_A_repre
            self.FQ1_repre_concat = tf.concat([tf.expand_dims(self.cov_pool_FQ1, 1), tf.expand_dims(self.x_batch_FQ1_img2, 1), tf.expand_dims(self.FQ1_tl_repres, 1)], 1)
            # FQ1_A(16,3,3)
            self.FQ1_text_img_sim = tf.expand_dims(self.cosine_distance(self.cov_pool_FQ1, self.x_batch_FQ1_img2), -1) 
            self.FQ1_text_tl_sim = tf.expand_dims(self.cosine_distance(self.cov_pool_FQ1, self.FQ1_tl_repres), -1)    
            self.FQ1_Img_tl_sim = tf.expand_dims(self.cosine_distance(self.x_batch_FQ1_img2, self.FQ1_tl_repres), -1)
            self.FQ1_A_text = tf.expand_dims(tf.concat([self.TW_self, self.FQ1_text_img_sim, self.FQ1_text_tl_sim], -1),1)
            self.FQ1_A_img = tf.expand_dims(tf.concat([self.FQ1_text_img_sim, self.TW_self, self.FQ1_Img_tl_sim], -1),1)
            self.FQ1_A_tl = tf.expand_dims(tf.concat([self.FQ1_text_tl_sim, self.FQ1_Img_tl_sim, self.TW_self], -1),1)
            self.FQ1_A = tf.concat([self.FQ1_A_text, self.FQ1_A_img, self.FQ1_A_tl], 1)
            # FQ1_A_GCN
            self.FQ1_gc1_out = self.GraphConvolution(self.FQ1_repre_concat, 1024, 512, self.FQ1_A, name='gener_FQ1_gc1')
            self.FQ1_gc2_out = self.GraphConvolution(self.FQ1_gc1_out, 512, 200, self.FQ1_A, name='gener_FQ1_gc2')

            # FQ2_A_repre
            self.FQ2_repre_concat = tf.concat([tf.expand_dims(self.cov_pool_FQ2, 1), tf.expand_dims(self.x_batch_FQ2_img2, 1), tf.expand_dims(self.FQ2_tl_repres, 1)], 1)
            # FQ2_A(16,3,3)
            self.FQ2_text_img_sim = tf.expand_dims(self.cosine_distance(self.cov_pool_FQ2, self.x_batch_FQ2_img2), -1) 
            self.FQ2_text_tl_sim = tf.expand_dims(self.cosine_distance(self.cov_pool_FQ2, self.FQ2_tl_repres), -1)    
            self.FQ2_Img_tl_sim = tf.expand_dims(self.cosine_distance(self.x_batch_FQ2_img2, self.FQ2_tl_repres), -1)
            self.FQ2_A_text = tf.expand_dims(tf.concat([self.TW_self, self.FQ2_text_img_sim, self.FQ2_text_tl_sim], -1),1)
            self.FQ2_A_img = tf.expand_dims(tf.concat([self.FQ2_text_img_sim, self.TW_self, self.FQ2_Img_tl_sim], -1),1)
            self.FQ2_A_tl = tf.expand_dims(tf.concat([self.FQ2_text_tl_sim, self.FQ2_Img_tl_sim, self.TW_self], -1),1)
            self.FQ2_A = tf.concat([self.FQ2_A_text, self.FQ2_A_img, self.FQ2_A_tl], 1)
            # FQ2_A_GCN
            self.FQ2_gc1_out = self.GraphConvolution(self.FQ2_repre_concat, 1024, 512, self.FQ2_A, name='gener_FQ2_gc1')
            self.FQ2_gc2_out = self.GraphConvolution(self.FQ2_gc1_out, 512, 200, self.FQ2_A, name='gener_FQ2_gc2')

            # view_representation
            self.TW_view_repre = tf.reduce_mean(self.TW_gc2_out, axis=1) 
            self.FQ1_view_repre = tf.reduce_mean(self.FQ1_gc2_out, axis=1)
            self.FQ2_view_repre = tf.reduce_mean(self.FQ2_gc2_out, axis=1)

            # 1024
            self.TW1_network_repres = tf.nn.embedding_lookup(node_embedding_TW1, self.index_tw1)
            self.FQ1_network_repres = tf.nn.embedding_lookup(node_embedding_FQ1, self.index_fq1)
            self.FQ2_network_repres = tf.nn.embedding_lookup(node_embedding_FQ1, self.index_fq2)


            self.TW1_con_rep1 = tf.add(self.TW_view_repre, self.TW1_network_repres)
            self.FQ1_con_rep1 = tf.add(self.FQ1_view_repre, self.FQ1_network_repres)
            self.FQ2_con_rep1 = tf.add(self.FQ2_view_repre, self.FQ2_network_repres)

            # tw
            index_tw1_ = tf.expand_dims(self.index_tw1, -1)
            self.node_embedding_TW_model = tf.scatter_nd_update(node_embedding_TW1, index_tw1_, self.TW1_con_rep1) # 更新之后每个user的表示
            self.update_TW = tf.expand_dims(self.node_embedding_TW_model, 0)

            # fq1
            index_fq1_ = tf.expand_dims(self.index_fq1, -1)
            node_embedding_FQ_model = tf.scatter_nd_update(node_embedding_FQ1, index_fq1_, self.FQ1_con_rep1)   
            # fq2
            index_fq2_ = tf.expand_dims(self.index_fq2, -1)
            node_embedding_FQ_model = tf.scatter_nd_update(node_embedding_FQ_model, index_fq2_, self.FQ2_con_rep1)
            self.update_FQ2 = tf.expand_dims(node_embedding_FQ_model,0)

            h_1_fc_TW, self.coefs_TW = self.node_repre(self.update_TW, n_heads, biases_tw, name='gener_gat_tw')
            h_1_fc_FQ, self.coefs_FQ = self.node_repre(self.update_FQ2, n_heads, biases_fq,  name='gener_gat_fq')

            TW_node_repres = tf.squeeze(h_1_fc_TW)
            FQ_node_repres = tf.squeeze(h_1_fc_FQ)
            self.TW_node_repres = TW_node_repres
            self.FQ_node_repres = FQ_node_repres

            # 1024
            self.TW1_repres = tf.nn.embedding_lookup(TW_node_repres, self.index_tw1) # 邻居的影响表示，没有考虑自身的表示
            self.FQ1_repres = tf.nn.embedding_lookup(FQ_node_repres, self.index_fq1)
            self.FQ2_repres = tf.nn.embedding_lookup(FQ_node_repres, self.index_fq2)

            initializer1 = tf.random_uniform_initializer(-1, 1)
            initializer2 = tf.random_uniform_initializer(-1, 1)

            self.Weight1 = tf.get_variable(name='gener_weight1',shape=[con_rep_di, bits],dtype=tf.float32, initializer=initializer1)
            self.bias1 = tf.get_variable(name='gener_bias1', shape=[bits], dtype=tf.float32, initializer=initializer1)
            self.Weight2 = tf.get_variable(name='gener_weight2',shape=[con_rep_di, bits],dtype=tf.float32, initializer=initializer2)
            self.bias2 = tf.get_variable(name='gener_bias2', shape=[bits],dtype=tf.float32, initializer=initializer2)

            t1_ = tf.nn.leaky_relu(tf.matmul(self.TW1_repres, self.Weight1) + self.bias1)
            self.t1_ = tf.nn.dropout(t1_, self.keep_prob)

            # 64, 128
            f1_ = tf.nn.leaky_relu(tf.matmul(self.FQ1_repres, self.Weight2) + self.bias2) 
            self.f1_ = tf.nn.dropout(f1_, self.keep_prob)

            f2_ = tf.nn.leaky_relu(tf.matmul(self.FQ2_repres, self.Weight2) + self.bias2)
            self.f2_ = tf.nn.dropout(f2_, self.keep_prob)            

            self.bpr_loss()

        def GraphConvolution(self, input_2, in_dim, out_dim, A_matrix, name):
            with tf.variable_scope(name):
                support = tf.layers.dense(input_2, out_dim, activation=None, name='weight', use_bias=False)
                output = tf.nn.leaky_relu(tf.matmul(A_matrix, support))
            return output


        def cosine_distance(self, x1, x2):
            cosin1 = tf.reduce_mean(tf.multiply(x1, x2), axis=1)
            return cosin1

        def att_repre2(self, own_rep, friend_rep, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                r_context = tf.Variable(tf.truncated_normal([node_feature]), name='r_context')
                h1 = layers.fully_connected(own_rep, node_feature, activation_fn=tf.nn.tanh)  # own_representation
                h2 = layers.fully_connected(friend_rep, node_feature, activation_fn=tf.nn.tanh)  # friend_representation

                sum_h1 = tf.reduce_sum(tf.multiply(h1, r_context), axis=1, keep_dims=True)  # (16, 1)
                sum_h2 = tf.reduce_sum(tf.multiply(h2, r_context), axis=1, keep_dims=True)

                c =tf.concat([sum_h1, sum_h2], 1)  # 16,2
                alpha = tf.nn.softmax(c, dim=1)  # 16,2
                own_all_tensor =alpha[:,0]
                friend_all_tensor = alpha[:,1]
                inputs_own = own_rep*own_all_tensor[:, None]
                inputs_friend = friend_rep*friend_all_tensor[:, None]
                output_attention = tf.add(inputs_own, inputs_friend)

            return output_attention, alpha


        def att_repre(self, text_rep, visual_rep, tl_rep, social_rep, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                attention1 = tf.add(text_rep, visual_rep)
                attention2 = tf.add(attention1, tl_rep)
                attention3 = tf.add(attention2, social_rep)
                output_attention =  tf.layers.dense(attention3, node_feature, activation = nonlinearity)
            return output_attention

        def bpr_loss(self):
            self.f1_di = tf.expand_dims(self.f1_, 1) 
            self.f2_di = tf.expand_dims(self.f2_, 1) 
            self.f1_2_3 = tf.concat([self.f1_di, self.f2_di], 1)

            self.label_pre_di = tf.expand_dims(self.y_label_pre, -1)
            self.label_pre_di_ = tf.tile(self.label_pre_di, [1, 1, 128])
            self.f1_2_3_ = tf.multiply(self.f1_2_3, self.label_pre_di_)
            self.f1_2_3__ = tf.reduce_sum(self.f1_2_3_, axis=1) 
            self.t1_f123 = tf.reduce_mean(tf.multiply(self.t1_, self.f1_2_3__),1,keep_dims=True)


            self.a1 = tf.sigmoid(self.t1_f123)
            self.a = self.a1

            self.b = tf.log(self.a+1e-10)
            self.loss_bpr1 = -150*tf.reduce_mean(self.b)
        
            
            self.tw_class = self.domain_classifier(self.t1_, self.l, name='adver_dc')
            self.label_di = tf.expand_dims(self.y_label, -1)
            self.label_di_ = tf.tile(self.label_di, [1, 1, 128])
            self.f1_2_3_label = tf.multiply(self.f1_2_3, self.label_di_)
            self.fq_repre = tf.reduce_sum(self.f1_2_3_label, axis=1) 
            self.fq_class = self.domain_classifier(self.fq_repre, self.l, name='adver_dc', reuse=True)

            self.domain_class_loss = 30*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.tw_class, labels=tf.cast(self.x_batch_TW_single, tf.float32)) + tf.nn.softmax_cross_entropy_with_logits(logits=self.fq_class, labels=tf.cast(self.x_batch_FQ_single, tf.float32))))

            self.cost_both = self.domain_class_loss 

            self.t_vars = tf.trainable_variables()    
            self.loss_norm = 0.0001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in self.t_vars ]) 
            self.vf_vars = [v for v in self.t_vars if 'gener_' in v.name]
            self.dc_vars = [v for v in self.t_vars if 'adver_' in v.name]

            self.emb_train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.5).minimize(self.loss_bpr1, var_list=self.vf_vars)
            self.domain_train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_view,
                beta1=0.5).minimize(self.cost_both, var_list=self.dc_vars)

            self.t1f1 = tf.reduce_mean(tf.multiply(self.t1_, self.f1_), 1, keep_dims= True)
            self.t1f2 = tf.reduce_mean(tf.multiply(self.t1_, self.f2_), 1,  keep_dims= True)
            self.pre = tf.nn.softmax(tf.concat([self.t1f1, self.t1f2], 1), axis=1, name='predict') 


        def modal_adv(self):
            self.TW_text_class = self.view_classifier(self.cov_pool_TW1, self.l, name='adver_vc_fc_tw')
            self.FQ1_text_class = self.view_classifier(self.cov_pool_FQ1, self.l, name='adver_vc_fc_fq')
            self.FQ2_text_class = self.view_classifier(self.cov_pool_FQ2, self.l, name='adver_vc_fc_fq', reuse=True)

            self.TW_img_class = self.view_classifier(self.x_batch_TW1_img2, self.l, name='adver_vc_fc_tw', reuse=True)
            self.FQ1_img_class = self.view_classifier(self.x_batch_FQ1_img2, self.l, name='adver_vc_fc_fq', reuse=True)
            self.FQ2_img_class = self.view_classifier(self.x_batch_FQ2_img2, self.l, name='adver_vc_fc_fq', reuse=True)

            self.TW_tl_class = self.view_classifier(self.TW_tl_repres, self.l, name='adver_vc_fc_tw', reuse=True)
            self.FQ1_tl_class = self.view_classifier(self.FQ1_tl_repres, self.l, name='adver_vc_fc_fq', reuse=True)
            self.FQ2_tl_class = self.view_classifier(self.FQ2_tl_repres, self.l, name='adver_vc_fc_fq', reuse=True)

            self.TW_social_class = self.view_classifier(self.TW1_network_repres, self.l, name='adver_vc_fc_tw', reuse=True)
            self.FQ1_social_class = self.view_classifier(self.FQ1_network_repres, self.l, name='adver_vc_fc_fq', reuse=True)
            self.FQ2_social_class = self.view_classifier(self.FQ2_network_repres, self.l, name='adver_vc_fc_fq', reuse=True)

            self.label_text = tf.concat([tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, 1]), tf.zeros([self.batch_size, 1]), tf.zeros([self.batch_size, 1])], 1)
            self.label_img = tf.concat([tf.zeros([self.batch_size, 1]), tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, 1]), tf.zeros([self.batch_size, 1])], 1)
            self.label_tl = tf.concat([tf.zeros([self.batch_size, 1]), tf.zeros([self.batch_size, 1]), tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, 1])], 1)
            self.label_social = tf.concat([tf.zeros([self.batch_size, 1]), tf.zeros([self.batch_size, 1]), tf.zeros([self.batch_size, 1]), tf.ones([self.batch_size, 1])], 1)

      
            self.view_class_loss_tw =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.TW_text_class, labels=self.label_text)+tf.nn.softmax_cross_entropy_with_logits(logits=self.TW_img_class, labels=self.label_img)+tf.nn.softmax_cross_entropy_with_logits(logits=self.TW_tl_class, labels=self.label_tl)+tf.nn.softmax_cross_entropy_with_logits(logits=self.TW_social_class, labels=self.label_social))
            
            self.view_class_loss_fq1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ1_text_class, labels=self.label_text)+tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ1_img_class, labels=self.label_img)+tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ1_tl_class, labels=self.label_tl)+tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ1_social_class, labels=self.label_social))

            self.view_class_loss_fq2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ2_text_class, labels=self.label_text)+tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ2_img_class, labels=self.label_img)+tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ2_tl_class, labels=self.label_tl)+tf.nn.softmax_cross_entropy_with_logits(logits=self.FQ2_social_class, labels=self.label_social))
     
            self.view_vlass_loss = 30*(tf.reduce_mean(self.view_class_loss_tw + self.view_class_loss_fq1 + self.view_class_loss_fq2))


        def view_classifier(self, repre, l, name, reuse=False): 
            with tf.variable_scope(name, reuse=reuse):
                repre = flip_gradient(repre, l)
                view_cla1 = tf.layers.dense(repre, 512, activation=None, name='view_fc_1')
                view_cla2 = tf.layers.dense(view_cla1, 256, activation=None, name='view_fc_2')
                view_cla3 = tf.layers.dense(view_cla2, 4, activation=None, name='view_fc_3')
            return view_cla3

        def extract_tl(self, tl_matrix, hidden_size, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                GRU_cell = rnn.GRUCell(hidden_size)
                GRU_cell = tf.nn.rnn_cell.DropoutWrapper(GRU_cell, 0.5) 
                output, final_state = tf.nn.dynamic_rnn(cell=GRU_cell, inputs=tl_matrix, sequence_length=length(tl_matrix),dtype=tf.float32)      
                tl_repre_ = tf.reduce_mean(output, axis=1)
                tl_repre = tf.layers.dense(tl_repre_, 1024, activation=nonlinearity)               
            return tl_repre 

        def domain_classifier(self, repre, l, name, reuse=False): 
            with tf.variable_scope(name, reuse=reuse):
                repre = flip_gradient(repre, l)
                do_cla1 = tf.layers.dense(repre, 64, activation=None, name='adver_domain_fc1')
                do_cla2 = tf.layers.dense(do_cla1, 32, activation=None, name='adver_domain_fc2')
                do_cla3 = tf.layers.dense(do_cla2, 2, activation=None, name='adver_domain_fc3')
            return do_cla3


        def node_repre(self, node_embedding, n_heads, biases_media, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                attns = []
                for _ in range(n_heads[0]):
                    seq_fts = tf.layers.conv1d(node_embedding, 512, 1, use_bias=False) 
                    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
                    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
                    logits = f_1 + tf.transpose(f_2, [0, 2, 1]) 
                    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + biases_media)  
                    vals = tf.matmul(coefs, seq_fts)
                    attns.append(vals)
                h_1 = tf.concat(attns, axis=-1)
                h_1_fc = tf.layers.dense(h_1, node_feature, activation=nonlinearity) 
            return h_1_fc, coefs
            
        def extract_content(self, input_x, num_filters, kernel_size, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                cov_x = tf.layers.conv1d(input_x, num_filters, kernel_size)
                cov_pool = tf.reduce_mean(cov_x, axis=2)
                cov_out = tf.layers.dense(cov_pool, 1024, activation=nonlinearity)

            return cov_out

        def extract_img(self, input_x, output_dim, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                output_x = tf.layers.dense(input_x, output_dim, activation = nonlinearity)
            return output_x

        def extract_social(self, input_x, output_dim, name, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                output_x = tf.layers.dense(input_x, output_dim, activation = nonlinearity)
            return output_x


    model = MODEL(learning_rate)
    print('---------Model Initialized!-------------')

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()    

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run([node_embedding_TW1, node_embedding_FQ1])

        for epoch in range(num_epochs):
            p = float(epoch) / num_epochs
            l = 2. / (1. + np.exp(-10. * p)) - 1

            batch = 0
            starttime_train = datetime.datetime.now()
            for i in range(0, len(train_user), batch_size):   
                batch = batch + 1
                x_index_batch = train_user[i: i+batch_size]
                y_index_batch = train_label[i: i+batch_size]
                y_index_re_batch = label_lines_re[i: i+batch_size]
                x_tw_single_batch = label_tw_single_train[i: i+batch_size]
                x_fq_single_batch = label_fq_single_train[i:i+batch_size]

                x_batch_TW1, x_batch_FQ1, x_batch_FQ2= load_content(x_index_batch)

                x_img_TW1, x_img_FQ1, x_img_FQ2 = load_img(x_index_batch)

                tw1, fq1, fq2 = index_dict(x_index_batch, tw_index_lines, fq_index_lines) 

                x_tl_TW1, x_tl_FQ1, x_tl_FQ2 = load_tl(x_index_batch)


                # discriminator
                loss, _, _, pre, loss_bpr1, loss_norm, domain_class_loss, TW1_con_rep1, coefs_TW = sess.run(
                    [model.cost_both, model.emb_train_op, model.domain_train_op, model.pre, model.loss_bpr1,
                     model.loss_norm, model.domain_class_loss, model.TW1_con_rep1, model.coefs_TW],
                    feed_dict={model.x_batch_TW1: x_batch_TW1, model.x_batch_FQ1: x_batch_FQ1,
                               model.x_batch_FQ2: x_batch_FQ2, model.x_batch_TW1_img: x_img_TW1,
                               model.x_batch_FQ1_img: x_img_FQ1, model.x_batch_FQ2_img: x_img_FQ2,
                               model.index_tw1: tw1, model.index_fq1: fq1, model.index_fq2: fq2,
                               model.batch_size: len(x_index_batch), model.y_label: y_index_batch,
                               model.y_label_pre: y_index_re_batch, model.training: True, model.keep_prob: 0.5,
                               model.x_batch_TW_single: x_tw_single_batch, model.x_batch_FQ_single: x_fq_single_batch,
                               model.x_batch_TW1_tl: x_tl_TW1, model.x_batch_FQ1_tl: x_tl_FQ1,
                               model.x_batch_FQ2_tl: x_tl_FQ2, model.l: l})

                y_true = y_index_batch.reshape(-1)
                y_pred = pre.reshape(-1)   

                y_true_index = np.argmax(y_index_batch, 1)
                y_pred_index = np.argmax(pre, 1)

                # auc
                auc_score = roc_auc_score(y_true, y_pred)
                # precision
                precision = precision_score(y_true_index, y_pred_index)
                precision1 = precision_score(y_true_index, y_pred_index)
                # recall
                recall = recall_score(y_true_index, y_pred_index)
                # F1
                f1 = f1_score(y_true_index, y_pred_index)
                # accuracy
                accuracy = accuracy_score(y_true_index, y_pred_index)

                print('-Lamda: %f, ----epoch: %s,----batch: %s-----loss: %f,----cost_bpr: %f,----cost_norm: %f------domain_class_loss: %f' % (lamda, str(epoch), str(batch), loss, loss_bpr1, loss_norm, domain_class_loss)) 

                # two discriminators
                f_loss = open('./result_adjust/UIL2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/loss_%s_train_5_3_5_discriminator.txt' % (str(lamda)),'a')
                f_loss.write(str(loss)+'\n')
                f_loss.close()
                # generator
                f_loss_bpr = open('./result_adjust/UIL2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/loss_%s_train_5_3_5_generator.txt' % (str(lamda)),'a')
                f_loss_bpr.write(str(loss_bpr1)+'\n')
                f_loss_bpr.close()

                if batch == 1:
                    epoch_pred = pre
                    epoch_true = y_index_batch
                    epoch_true_index = y_true_index
                    epoch_pred_index = y_pred_index

                else:
                    epoch_pred = np.concatenate([epoch_pred, pre], axis=0)
                    epoch_true = np.concatenate([epoch_true, y_index_batch], axis=0)
                    epoch_true_index = np.concatenate([epoch_true_index, y_true_index], axis=0)      
                    epoch_pred_index = np.concatenate([epoch_pred_index, y_pred_index], axis=0)    

            epoch_y_true = np.reshape(epoch_true, [-1])
            epoch_y_pred = np.reshape(epoch_pred, [-1])
            epoch_y_true_index = np.reshape(epoch_true_index, [-1])
            epoch_y_pred_index = np.reshape(epoch_pred_index, [-1])

            # epoch_metric
            auc_score_train = roc_auc_score(epoch_y_true, epoch_y_pred)
            precision_train = precision_score(epoch_y_true_index, epoch_y_pred_index)
            recall_train = recall_score(epoch_y_true_index, epoch_y_pred_index)
            f1_train = f1_score(epoch_y_true_index, epoch_y_pred_index)
            accuracy_train = accuracy_score(epoch_y_true_index, epoch_y_pred_index)

            endtime_train = datetime.datetime.now()
            
            print('-Lamda: %f, *****epoch: %s,*****training time: %f*************auc: %f------precision: %f------recall: %f------f1: %f------accuracy: %f' % (lamda, str(epoch), (endtime_train - starttime_train).seconds, auc_score_train, precision_train, recall_train, f1_train, accuracy_train))  
            f_train1_auc = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/auc_%s_train_5_3_5.txt' % (str(lamda)),'a')
            f_train1_auc.write(str(auc_score_train)+'\n')
            f_train1_auc.close()  

            f_train1_precision = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/precision_%s_train_5_3_5.txt' % (str(lamda)),'a')
            f_train1_precision.write(str(precision_train)+'\n')
            f_train1_precision.close()     

            f_train1_recall = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/recall_%s_train_5_3_5.txt' % (str(lamda)),'a')
            f_train1_recall.write(str(recall_train)+'\n')
            f_train1_recall.close()        

            f_train1_f1 = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/f1_%s_train_5_3_5.txt' % (str(lamda)),'a')
            f_train1_f1.write(str(f1_train)+'\n')
            f_train1_f1.close()                

            f_train1_accuracy = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/accuracy_%s_train_5_3_5.txt' % (str(lamda)),'a')
            f_train1_accuracy.write(str(accuracy_train)+'\n')
            f_train1_accuracy.close()                                                
            
            batch_test = 0
            starttime_test = datetime.datetime.now()   
            for j in range(0, len(test_user), batch_size):  
                batch_test = batch_test + 1
                x_index_batch_test = test_user[j: j+batch_size]
                y_index_batch_test = test_label[j: j+batch_size]
                y_index_re_batch_test = label_lines_re_test[j: j+batch_size]
                x_tw_single_batch_test = label_tw_single_test[j: j+batch_size]
                x_fq_single_batch_test = label_fq_single_test[j: j+batch_size]


                x_batch_TW1_test, x_batch_FQ1_test, x_batch_FQ2_test = load_content(x_index_batch_test)

                x_img_TW1_test, x_img_FQ1_test, x_img_FQ2_test = load_img(x_index_batch_test)

                tw1_test, fq1_test, fq2_test = index_dict(x_index_batch_test, tw_index_lines, fq_index_lines)

                x_tl_TW1_test, x_tl_FQ1_test, x_tl_FQ2_test = load_tl(x_index_batch_test)


                loss_test, pre_test, TW_node_repres, FQ_node_repres, Weight1, bias1, Weight2, bias2 = sess.run(
                    [model.cost_both, model.pre, model.TW_node_repres, model.FQ_node_repres,
                     model.Weight1, model.bias1, model.Weight2, model.bias2],
                    feed_dict={model.x_batch_TW1: x_batch_TW1_test,
                               model.x_batch_FQ1: x_batch_FQ1_test,
                               model.x_batch_FQ2: x_batch_FQ2_test,
                               model.index_tw1: tw1_test, model.index_fq1: fq1_test,
                               model.index_fq2: fq2_test,
                               model.batch_size: len(x_index_batch_test),
                               model.y_label: y_index_batch_test,
                               model.y_label_pre: y_index_re_batch_test,
                               model.training: False, model.keep_prob: 1,
                               model.x_batch_TW1_img: x_img_TW1_test,
                               model.x_batch_FQ1_img: x_img_FQ1_test,
                               model.x_batch_FQ2_img: x_img_FQ2_test,
                               model.x_batch_TW_single: x_tw_single_batch_test,
                               model.x_batch_FQ_single: x_fq_single_batch_test,
                               model.x_batch_TW1_tl: x_tl_TW1_test,
                               model.x_batch_FQ1_tl: x_tl_FQ1_test,
                               model.x_batch_FQ2_tl: x_tl_FQ2_test})
                np.save('update_TW.npy', TW_node_repres)
                np.save('update_FQ2.npy', FQ_node_repres)
                np.save('Weight1.npy', Weight1)
                np.save('bias1.npy', bias1)
                np.save('Weight2.npy', Weight2)
                np.save('bias2.npy', bias2)

                y_true_test = y_index_batch_test.reshape(-1)
                y_pred_test = pre_test.reshape(-1)  

                y_true_index_test = np.argmax(y_index_batch_test, 1)
                y_pred_index_test = np.argmax(pre_test, 1)

                auc_score_test = roc_auc_score(y_true_test, y_pred_test)                              
                precision_test = precision_score(y_true_index_test, y_pred_index_test)
                recall_test = recall_score(y_true_index_test, y_pred_index_test)
                f1_test = f1_score(y_true_index_test, y_pred_index_test)
                accuracy_test = accuracy_score(y_true_index_test, y_pred_index_test)


                print('-Lamda: %f, ----epoch: %s,----batch: %s-----loss_validate: %f,-------auc_validate: %f------precision_validate: %f------recall_validate: %f------f1_validate: %f------accuracy_validate: %f' % (lamda, str(epoch), str(batch_test), loss_test,  auc_score_test, precision_test, recall_test, f1_test, accuracy_test))

                
                if batch_test == 1:
                    epoch_pred_test = pre_test
                    epoch_true_test = y_index_batch_test
                    epoch_true_index_test = y_true_index_test
                    epoch_pred_index_test = y_pred_index_test
                else:
                    epoch_pred_test = np.concatenate([epoch_pred_test, pre_test], axis=0)
                    epoch_true_test = np.concatenate([epoch_true_test, y_index_batch_test], axis=0)
                    epoch_true_index_test = np.concatenate([epoch_true_index_test, y_true_index_test], axis=0)      
                    epoch_pred_index_test = np.concatenate([epoch_pred_index_test, y_pred_index_test], axis=0)   

            epoch_y_true_test = np.reshape(epoch_true_test, [-1])
            epoch_y_pred_test = np.reshape(epoch_pred_test, [-1]) 
            epoch_y_true_index_test = np.reshape(epoch_true_index_test, [-1])
            epoch_y_pred_index_test = np.reshape(epoch_pred_index_test, [-1])

            auc_score_test_epoch = roc_auc_score(epoch_y_true_test, epoch_y_pred_test)
            precision_test_epoch = precision_score(epoch_y_true_index_test, epoch_y_pred_index_test)
            recall_test_epoch = recall_score(epoch_y_true_index_test, epoch_y_pred_index_test)
            f1_test_epoch = f1_score(epoch_y_true_index_test, epoch_y_pred_index_test)
            accuracy_test_epoch = accuracy_score(epoch_y_true_index_test, epoch_y_pred_index_test)            

            endtime_test = datetime.datetime.now()            


            f_test_auc = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/auc_%s_validate_5_3_5.txt' % (str(lamda)),'a')
            f_test_auc.write(str(auc_score_test_epoch)+'\n')
            f_test_auc.close()  

            f_test_precision = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/precision_%s_validate_5_3_5.txt' % (str(lamda)),'a')
            f_test_precision.write(str(precision_test_epoch)+'\n')
            f_test_precision.close()     

            f_test_recall = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/recall_%s_validate_5_3_5.txt' % (str(lamda)),'a')
            f_test_recall.write(str(recall_test_epoch)+'\n')
            f_test_recall.close()        

            f_test_f1 = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/f1_%s_validate_5_3_5.txt' % (str(lamda)),'a')
            f_test_f1.write(str(f1_test_epoch)+'\n')
            f_test_f1.close()                

            f_test_accuracy = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/accuracy_%s_validate_5_3_5.txt' % (str(lamda)),'a')
            f_test_accuracy.write(str(accuracy_test_epoch)+'\n')
            f_test_accuracy.close()    


            print('-Lamda: %f, *****epoch: %s,*****validate time: %f*************auc: %f------precision: %f------recall: %f------f1: %f------accuracy: %f' % (lamda, str(epoch), (endtime_test - starttime_test).seconds, auc_score_test_epoch, precision_test_epoch, recall_test_epoch, f1_test_epoch, accuracy_test_epoch))  

            # Real_test

            batch_test2 = 0
            starttime_test2 = datetime.datetime.now()  
            for k in range(0, len(test_user2), batch_size):
                batch_test2 = batch_test2 + 1
                x_index_all_test = test_user2[k: k+batch_size]
                y_index_all_test = test_label2[k: k+batch_size]
                y_index_re_all_test = label_lines_re_test2[k: k+batch_size]
                x_tw_single_all_test = label_tw_single_test2[k: k+batch_size]
                x_fq_single_all_test = label_fq_single_test2[k: k+batch_size]   

                x_all_TW1_test, x_all_FQ1_test, x_all_FQ2_test = load_content(x_index_all_test)

                x_all_img_TW1_test, x_all_img_FQ1_test, x_all_img_FQ2_test = load_img(x_index_all_test)

                tw1_all_test, fq1_all_test, fq2_all_test = index_dict(x_index_all_test, tw_index_lines, fq_index_lines)

                x_all_tl_TW1_test, x_all_tl_FQ1_test, x_all_tl_FQ2_test = load_tl(x_index_all_test) 

                pre_all_test, coefs_TW_test = sess.run([model.pre, model.coefs_TW], feed_dict={model.x_batch_TW1:x_all_TW1_test, model.x_batch_FQ1:x_all_FQ1_test, model.x_batch_FQ2:x_all_FQ2_test, model.index_tw1:tw1_all_test, model.index_fq1:fq1_all_test, model.index_fq2: fq2_all_test, model.batch_size:len(x_index_all_test), model.y_label:y_index_all_test, model.y_label_pre: y_index_re_all_test, model.training:False, model.keep_prob: 1, model.x_batch_TW1_img:x_all_img_TW1_test, model.x_batch_FQ1_img: x_all_img_FQ1_test, model.x_batch_FQ2_img: x_all_img_FQ2_test,  model.x_batch_TW_single:x_tw_single_all_test, model.x_batch_FQ_single:x_fq_single_all_test, model.x_batch_TW1_tl:x_all_tl_TW1_test, model.x_batch_FQ1_tl:x_all_tl_FQ1_test, model.x_batch_FQ2_tl:x_all_tl_FQ2_test}) 

                y_true_test2 = y_index_all_test.reshape(-1)
                y_pred_test2 = pre_all_test.reshape(-1)  

                y_true_index_test2 = np.argmax(y_index_all_test, 1)
                y_pred_index_test2 = np.argmax(pre_all_test, 1)

                auc_score_test2 = roc_auc_score(y_true_test2, y_pred_test2)
                precision_test2 = precision_score(y_true_index_test2, y_pred_index_test2)
                recall_test2 = recall_score(y_true_index_test2, y_pred_index_test2)
                f1_test2 = f1_score(y_true_index_test2, y_pred_index_test2)
                accuracy_test2 = accuracy_score(y_true_index_test2, y_pred_index_test2) 

                print('-Lamda: %f, ----epoch: %s,----batch: %s------------auc_test: %f------precision_test: %f------recall_test: %f------f1_test: %f------accuracy_test: %f' % (lamda, str(epoch), str(batch_test2),  auc_score_test2, precision_test2, recall_test2, f1_test2, accuracy_test2))

                if batch_test2 == 1:
                    epoch_pred_test2 = pre_all_test
                    epoch_true_test2 = y_index_all_test
                    epoch_true_index_test2 = y_true_index_test2
                    epoch_pred_index_test2 = y_pred_index_test2
                else:
                    epoch_pred_test2 = np.concatenate([epoch_pred_test2, pre_all_test], axis=0)
                    epoch_true_test2 = np.concatenate([epoch_true_test2, y_index_all_test], axis=0)
                    epoch_true_index_test2 = np.concatenate([epoch_true_index_test2, y_true_index_test2], axis=0)      
                    epoch_pred_index_test2 = np.concatenate([epoch_pred_index_test2, y_pred_index_test2], axis=0) 

            epoch_y_true_test2 = np.reshape(epoch_true_test2, [-1])
            epoch_y_pred_test2 = np.reshape(epoch_pred_test2, [-1])
            # print(epoch_y_true_test2)
            # print(epoch_y_pred_test2)
            epoch_y_true_index_test2 = np.reshape(epoch_true_index_test2, [-1])
            epoch_y_pred_index_test2 = np.reshape(epoch_pred_index_test2, [-1])

            auc_score_test_epoch2 = roc_auc_score(epoch_y_true_test2, epoch_y_pred_test2)
            precision_test_epoch2 = precision_score(epoch_y_true_index_test2, epoch_y_pred_index_test2)
            recall_test_epoch2 = recall_score(epoch_y_true_index_test2, epoch_y_pred_index_test2)
            f1_test_epoch2 = f1_score(epoch_y_true_index_test2, epoch_y_pred_index_test2)
            accuracy_test_epoch2 = accuracy_score(epoch_y_true_index_test2, epoch_y_pred_index_test2)            

            endtime_test2 = datetime.datetime.now()            

            f_test_auc = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/auc_%s_test_5_3_5.txt' % (str(lamda)),'a')
            f_test_auc.write(str(auc_score_test_epoch2)+'\n')
            f_test_auc.close()  

            f_test_precision = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/precision_%s_test_5_3_5.txt' % (str(lamda)),'a')
            f_test_precision.write(str(precision_test_epoch2)+'\n')
            f_test_precision.close()     

            f_test_recall = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/recall_%s_test_5_3_5.txt' % (str(lamda)),'a')
            f_test_recall.write(str(recall_test_epoch2)+'\n')
            f_test_recall.close()        

            f_test_f1 = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/f1_%s_test_5_3_5.txt' % (str(lamda)),'a')
            f_test_f1.write(str(f1_test_epoch2)+'\n')
            f_test_f1.close()                

            f_test_accuracy = open('./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Record/accuracy_%s_test_5_3_5.txt' % (str(lamda)),'a')
            f_test_accuracy.write(str(accuracy_test_epoch2)+'\n')
            f_test_accuracy.close()  

            print('-Lamda: %f, *****epoch: %s,*****test time: %f*************auc: %f------precision: %f------recall: %f------f1: %f------accuracy: %f' % (lamda, str(epoch), (endtime_test2 - starttime_test2).seconds, auc_score_test_epoch2, precision_test_epoch2, recall_test_epoch2, f1_test_epoch2, accuracy_test_epoch2))             

            if accuracy_test_epoch > max_acc:
                num_save_epoch2 = epoch
                max_auc = auc_score_test_epoch
                max_precision = precision_test_epoch
                max_recall = recall_test_epoch
                max_f1 = f1_test_epoch
                max_acc = accuracy_test_epoch  


            if accuracy_test_epoch2 > max_acc_test:
                save_path = saver.save(sess, model_path)
                num_save_epoch = epoch
                # test
                max_auc_test = auc_score_test_epoch2
                max_precision_test = precision_test_epoch2
                max_recall_test = recall_test_epoch2
                max_f1_test = f1_test_epoch2
                max_acc_test = accuracy_test_epoch2


        with open("./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Final/validate_final_result_%s_5_3_5.txt" % str(lamda), 'a') as ff1:
            ff1.write('lamda: %s\n' % (str(lamda)))
            ff1.write('num_save_epoch_from_0_validate: %s\n' % (str(num_save_epoch2)))
            ff1.write('max_auc: %s, max_precision: %s, max_recall: %s, max_f1: %s, max_acc: %s\n' % (str(max_auc), str(max_precision), str(max_recall), str(max_f1), str(max_acc)))
        with open("./result_adjust/MoToInv2_5/5_3_5_5_5_att_2_check_4_6_109_109_2_update_2/Final/test_final_result_%s_5_3_5.txt" % str(lamda), 'a') as ff1:
            ff1.write('lamda: %s\n' % (str(lamda)))
            ff1.write('num_save_epoch_from_0_test: %s\n' % (str(num_save_epoch)))
            ff1.write('max_auc: %s, max_precision: %s, max_recall: %s, max_f1: %s, max_acc: %s\n' % (str(max_auc_test), str(max_precision_test), str(max_recall_test), str(max_f1_test), str(max_acc_test)))            

        print('------Validate------max_auc: %f, max_precision: %f, max_recall: %f, max_f1: %f, max_acc: %f'% (max_auc, max_precision, max_recall, max_f1, max_acc))

        print('------Test------max_auc: %f, max_precision: %f, max_recall: %f, max_f1: %f, max_acc: %f'% (max_auc_test, max_precision_test, max_recall_test, max_f1_test, max_acc_test))
        max_auc = 0
        max_precision = 0
        max_recall = 0
        max_f1 = 0
        max_acc = 0
        num_save_epoch = 0
        max_acc_test = 0
        max_auc_test = 0
        max_precision_test = 0
        max_recall_test = 0
        max_f1_test = 0
        max_acc_test = 0 
        num_save_epoch2 = 0
