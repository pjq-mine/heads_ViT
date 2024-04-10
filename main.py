import numpy as np
import csv
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity


path_biases_tw = '/data/TWFQ/twitter_foursquare/Network/Network/Origin/tw_bias_matrix.npy'
path_biases_fq = '/data/TWFQ/twitter_foursquare/Network/Network/Origin/fq_bias_matrix.npy'
biases_tw = np.load(path_biases_tw)
biases_fq = np.load(path_biases_fq)

path_embedding_tw = '/data/TWFQ/twitter_foursquare/Network/Network/Origin/tw_sort.npy'
path_embedding_fq = '/data/TWFQ/twitter_foursquare/Network/Network/Origin/fq_sort.npy'
embedding_tw = np.load(path_embedding_tw)
embedding_fq = np.load(path_embedding_fq)


path_dir = '/data/TWFQ/twitter_foursquare/new_dataset2_shuffle/new_dataset2_shuffle/'

path_index_tw = '/data/TWFQ/twitter_foursquare/Network/Network/Origin/user_index_tw.npy'
path_index_fq = '/data/TWFQ/twitter_foursquare/Network/Network/Origin/user_index_fq.npy'

tw_index_lines = np.load(path_index_tw).tolist()
fq_index_lines = np.load(path_index_fq).tolist()

print('------index lines loaded!------')


def load_user(path_dir):

    user_path = path_dir + 'train_name_shuffle.csv'  # 857[tw, fq1, fq2]
    user_test_path = path_dir + 'test_name_shuffle.csv'

    # user
    f_user = open(user_path, 'r', encoding='utf-8')
    reader_user = csv.reader(f_user)
    user_lines = [row[0] for row in reader_user]
    loss_dict = ['alexjamesfitz', 'iamedia007', 'elliotj651', 'whizhouse', 'niketa', 'rickyrobinett', 'scottdreyfus',
                 'joekanakaraj', 'theory__', 'persiancowboy', 'libbydoodle', 'genepark', 'johnedetroit', 'djidlemind',
                 'gyrus', 'anytimefitness', 'afpa1', 'dberkowitz', 'iprincemb', 'cherylrice', 'amalucky', 'opajdara',
                 'andrewphelps', 'edgarprado', 'rushwan', 'razabegg', 'megan_calimbas', 'liscallelura', 'rtsnance',
                 'thomastowell', 'sears', '_dom', 'johngarcia', 'visitcedarhill', 'jtw90210', 'bravotv', 'dales777',
                 'andrewvest', 'pgatour', 'agoldfisher', 'nycgov'
                 ]  
    history = []
    for i in range(0, len(user_lines)):
        if user_lines[i] in loss_dict:
            history.append(i)
    print(history)
    user_lines = [user_lines[i] for i in range(0, len(user_lines), 1) if i not in history]  # 821[tw, fq1, fq2]

    f_user_test = open(user_test_path, 'r', encoding='utf-8')
    reader_user_test = csv.reader(f_user_test)
    user_lines_test = [row[0] for row in reader_user_test]
    test_loss_dict = ['jorenerene', 'andrewvest', 'rushwan', 'thomastowell', 'johngarcia', 'rtsnance',
                      'gyrus']  # 101[tw, fq1, fq2]
    test_history = []
    for i in range(0, len(user_lines_test)):
        if user_lines_test[i] in test_loss_dict:
            test_history.append(i)
    print(test_history)
    user_lines_test = [user_lines_test[i] for i in range(0, len(user_lines_test), 1) if
                       i not in test_history]  # 101

    return user_lines, user_lines_test


train_user_, test_user_ = load_user(path_dir)
print('------user pair loaded!')
print('train_user_: %s, test_user_: %s' % (len(train_user_), len(test_user_)))


def load_user_test(path_dir):
    user_test_path = path_dir + 'validate_name_shuffle.csv'

    f_user_test = open(user_test_path, 'r', encoding='utf-8')
    reader_user_test = csv.reader(f_user_test)
    user_lines_test = [row[0] for row in reader_user_test]
    test_loss_dict = ['visitcedarhill', 'pgatour', 'jtw90210', 'dales777', 'agoldfisher', 'bravotv', 'gyrus']
    f_history = []
    for i in range(0, len(user_lines_test)):
        if user_lines_test[i] in test_loss_dict:
            f_history.append(i)
    print(f_history)
    user_lines_test = [user_lines_test[i] for i in range(0, len(user_lines_test), 1) if
                       i not in f_history]

    return user_lines_test


test_user_2 = load_user_test(path_dir)
print('------test_user pair loaded!')
print('test_user_2: %s' % (len(test_user_2)))


def index_dict(index, index_dict_tw, index_dict_fq):
    tw1 = []
    fq = []
    for pair in index:
        tw1.extend([index_dict_tw[pair]])
        fq.extend([index_dict_fq[pair]])
    return tw1, fq



num_epochs = 400
batch_size = 16
node_num_tw = 3463
node_num_fq = 3833
node_feature = 200
num_tw = 1000
num_fq = 52
embedding_dim = 768
num_filters = 1024  # 64--128--256---512---1024
kernel_size = 7  # 2--3---5---7---9
hidden_size = 150
img_dim = 2048
bits = 128
con_rep_di = 200
n_heads = [5, 1]  # additional entry for the output layer [8,1]
nonlinearity = tf.nn.elu
num_time = 77
num_city = 947
zero_list_tl = [[0 for i in range(num_city)] for j in range(num_time)]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.reset_default_graph()
np.random.shuffle(train_user_)
np.random.shuffle(test_user_)
np.random.shuffle(test_user_2)


node_embedding_TW = tf.Variable(tf.constant(0, shape=[node_num_tw, node_feature], dtype=tf.float32),
                                    name='gener_node_embedding_tw', trainable=True)
node_embedding_TW1 = tf.assign(node_embedding_TW, embedding_tw)

node_embedding_FQ = tf.Variable(tf.constant(0, shape=[node_num_fq, node_feature], dtype=tf.float32),
                                name='gener_node_embedding_fq', trainable=True)
node_embedding_FQ1 = tf.assign(node_embedding_FQ, embedding_fq)


class MODEL():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

        self.tw_index_lines = tw_index_lines
        self.fq_index_lines = fq_index_lines

        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l = tf.placeholder(tf.float32, [])

        self.training = tf.placeholder(dtype=tf.bool, shape=(), name='training')

        self.x_batch_TW1 = tf.placeholder(tf.float32, [None, num_tw, embedding_dim], name='x_batch_TW1')
        self.x_batch_FQ1 = tf.placeholder(tf.float32, [None, num_fq, embedding_dim], name='x_batch_FQ1')

        self.x_batch_TW1_img = tf.placeholder(tf.float32, [None, img_dim], name='x_batch_TW1_img')
        self.x_batch_FQ1_img = tf.placeholder(tf.float32, [None, img_dim], name='x_batch_FQ1_img')

        self.x_batch_TW1_tl = tf.placeholder(tf.float32, [None, num_time, num_city], name='x_batch_TW1_tl')
        self.x_batch_FQ1_tl = tf.placeholder(tf.float32, [None, num_time, num_city], name='x_batch_FQ1_tl')

        self.index_tw1 = tf.placeholder(tf.int32, [None], name='index_tw1')
        self.index_fq1 = tf.placeholder(tf.int32, [None], name='index_fq1')

        self.TW1_network_repres = tf.nn.embedding_lookup(node_embedding_TW1, self.index_tw1)
        self.FQ1_network_repres = tf.nn.embedding_lookup(node_embedding_FQ1, self.index_fq1)

        initializer1 = tf.random_uniform_initializer(-1, 1)
        initializer2 = tf.random_uniform_initializer(-1, 1)

        self.Weight1 = tf.get_variable(name='gener_weight1', shape=[con_rep_di, bits], dtype=tf.float32,
                                       initializer=initializer1)  # 200 128
        self.bias1 = tf.get_variable(name='gener_bias1', shape=[bits], dtype=tf.float32, initializer=initializer1)
        self.Weight2 = tf.get_variable(name='gener_weight2', shape=[con_rep_di, bits], dtype=tf.float32,
                                       initializer=initializer2)
        self.bias2 = tf.get_variable(name='gener_bias2', shape=[bits], dtype=tf.float32, initializer=initializer2)

        t1_ = tf.nn.leaky_relu(tf.matmul(self.TW1_network_repres, self.Weight1) + self.bias1)
        self.t1_ = tf.nn.dropout(t1_, self.keep_prob)

        fq = tf.nn.leaky_relu(tf.matmul(self.FQ1_network_repres, self.Weight2) + self.bias2)
        self.fq = tf.nn.dropout(fq, self.keep_prob)

        self.node_embedding_TW1 = tf.matmul(node_embedding_TW1, self.Weight1) + self.bias1
        self.node_embedding_FQ1 = tf.matmul(node_embedding_FQ1, self.Weight2) + self.bias2

        self.bpr_loss()


    def bpr_loss(self):
        self.compa_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.t1_, axis=1),
                                                         tf.nn.l2_normalize(self.fq, axis=1), dim=1, reduction=tf.losses.Reduction.SUM)

        self.t_vars = tf.trainable_variables()
        self.vf_vars = [v for v in self.t_vars if 'gener_' in v.name]

        self.class_train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5).minimize(self.compa_loss, var_list=self.vf_vars)


learning_rate = 0.05


model = MODEL(learning_rate)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


path_twitter = '/data/TWFQ/twitter_foursquare/Feature/Twitter_padding/'
path_foursquare = '/data/TWFQ/twitter_foursquare/Feature/Fq_padding/'
path_tw_img = '/data/TWFQ/twitter_foursquare/Feature/TW/'
path_fq_img = '/data/TWFQ/twitter_foursquare/Feature/FQ/'
path_tw_tl = '/data/TWFQ/twitter_foursquare/Feature/TW_Time_location_matrix/'
path_fq_tl = '/data/TWFQ/twitter_foursquare/Feature/Fq_Time_location_matrix/'
path_tw_tl = '/data/TWFQ/twitter_foursquare/Feature/TW_Time_location_matrix/'
path_fq_tl = '/data/TWFQ/twitter_foursquare/Feature/Fq_Time_location_matrix/'
tl_tw_exist_list = os.listdir(path_tw_tl)
tl_fq_exist_list = os.listdir(path_fq_tl)



def load_content(x_index):
    x_batch_TW1 = []
    x_batch_FQ1 = []

    for pair in x_index:
        pair_list1 = []
        pair_list2 = []

        TW_1 = path_twitter + pair + '.npy'
        FQ_1 = path_foursquare + pair + '.npy'

        try:
            data_TW1 = np.load(TW_1)
        except:
            print('/Twitter_padding/' + pair + '.npy')
        try:
            data_FQ1 = np.load(FQ_1)
        except:
            print(FQ_1)

        try:
            pair_list1.extend(data_TW1)
            pair_list2.extend(data_FQ1)
        except:
            pass

        x_batch_TW1.append(pair_list1)
        x_batch_FQ1.append(pair_list2)

    return np.array(x_batch_TW1), np.array(x_batch_FQ1)


def load_img(x_index):
    x_batch_TW1 = []
    x_batch_FQ1 = []

    for pair in x_index:
        pair_list1 = []
        pair_list2 = []

        TW_1 = path_tw_img + pair + '.npy'
        FQ_1 = path_fq_img + pair + '.npy'

        data_TW1 = np.load(TW_1)
        data_FQ1 = np.load(FQ_1)

        pair_list1.extend(data_TW1)
        pair_list2.extend(data_FQ1)

        x_batch_TW1.append(pair_list1)
        x_batch_FQ1.append(pair_list2)

    return np.array(x_batch_TW1), np.array(x_batch_FQ1)


def load_tl(x_index):
    x_batch_TW1 = []
    x_batch_FQ1 = []

    for pair in x_index:
        pair_list1 = []
        pair_list2 = []

        TW_1_ = pair + '.npy'
        FQ_1_ = pair + '.npy'

        if TW_1_ in tl_tw_exist_list:
            TW_1 = path_tw_tl + TW_1_
            data_TW1 = np.load(TW_1)
        else:
            data_TW1 = zero_list_tl

        if FQ_1_ in tl_fq_exist_list:
            FQ_1 = path_fq_tl + FQ_1_
            data_FQ1 = np.load(FQ_1)
        else:
            data_FQ1 = zero_list_tl

        pair_list1.extend(data_TW1)
        pair_list2.extend(data_FQ1)

        x_batch_TW1.append(pair_list1)
        x_batch_FQ1.append(pair_list2)

    return np.array(x_batch_TW1), np.array(x_batch_FQ1)



with tf.Session(config=config) as sess:
    sess.run(init_op)
    sess.run([node_embedding_TW1, node_embedding_FQ1])

    for epoch in range(num_epochs):
        p = float(epoch) / num_epochs
        l = 2. / (1. + np.exp(-10. * p)) - 1

        batch = 0
        true_1 = 0
        true_5 = 0
        for i in range(0, len(train_user_), batch_size):
            batch = batch + 1
            x_index_batch = train_user_[i: i + batch_size]
            x_batch_TW1, x_batch_FQ1 = load_content(x_index_batch)
            x_img_TW1, x_img_FQ1 = load_img(x_index_batch)
            x_tl_TW1, x_tl_FQ1 = load_tl(x_index_batch)
            tw, fq = index_dict(x_index_batch, tw_index_lines, fq_index_lines)
            t1_, fq_, class_train_op, compa_loss, embedding_TW1, embedding_FQ1, t_vars = sess.run(
                [model.t1_, model.fq, model.class_train_op, model.compa_loss, model.node_embedding_TW1, model.node_embedding_FQ1, model.t_vars],
                feed_dict={model.x_batch_TW1: x_batch_TW1, model.x_batch_FQ1: x_batch_FQ1,
                           model.x_batch_TW1_img: x_img_TW1,
                           model.x_batch_FQ1_img: x_img_FQ1,
                           model.index_tw1: tw,
                           model.index_fq1: fq,
                           model.batch_size: len(x_index_batch), model.training: True, model.keep_prob: 0.3,
                           model.x_batch_TW1_tl: x_tl_TW1, model.x_batch_FQ1_tl: x_tl_FQ1,
                           model.l: l})
            print(compa_loss)
            s = cosine_similarity(embedding_TW1, embedding_FQ1)
            sort = np.argsort(-s)
            pre_1 = sort[:, 0:1].tolist() #3833
            pre_5 = sort[:, 0:5].tolist()  # 3833
            j = 0
            for tw_i in tw:
                if fq[j] in pre_1[tw_i]:
                    true_1 += 1
                if fq[j] in pre_5[tw_i]:
                    true_5 += 1
                j += 1
        print("true@1==", true_1, "true@5==", true_5, "len_train", len(train_user_))
        print("ACC@1==", true_1/len(train_user_), "ACC@5==", true_5/len(train_user_))

        true_1 = 0
        true_5 = 0
        for i in range(0, len(test_user_), batch_size):
            batch = batch + 1
            x_index_batch = test_user_[i: i + batch_size]
            tw, fq = index_dict(x_index_batch, tw_index_lines, fq_index_lines)
            j = 0
            for tw_i in tw:
                if fq[j] in pre_1[tw_i]:
                    true_1 += 1
                if fq[j] in pre_5[tw_i]:
                    true_5 += 1
                j += 1
        print("vild_true@1==", true_1, "vild_true@5==", true_5, "len_vild", len(test_user_))
        print("vild_ACC@1==", true_1 / len(test_user_), "vild_ACC@5==", true_5 / len(test_user_))

        true_1 = 0
        true_5 = 0
        for i in range(0, len(test_user_2), batch_size):
            batch = batch + 1
            x_index_batch = test_user_2[i: i + batch_size]
            tw, fq = index_dict(x_index_batch, tw_index_lines, fq_index_lines)
            j = 0
            for tw_i in tw:
                if fq[j] in pre_1[tw_i]:
                    true_1 += 1
                if fq[j] in pre_5[tw_i]:
                    true_5 += 1
                j += 1
        print("test_true@1==", true_1, "test_true@5==", true_5, "len_test", len(test_user_2))
        print("test_ACC@1==", true_1 / len(test_user_2), "test_ACC@5==", true_5 / len(test_user_2))
