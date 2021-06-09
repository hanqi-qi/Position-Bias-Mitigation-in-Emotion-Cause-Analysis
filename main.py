# -*- encoding:utf-8 -*-
'''
@time: 2020/06/
@author: hanqiyan
@email: hanqi.yan@warwick.ac.uk
'''

import numpy as np
import pickle as pk
import transformer_layers as trans
import tensorflow as tf
import sys, os, time, codecs, pdb
import utils.model_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
import math
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# embedding parameters ##
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
# input shape ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_path_num', 10, 'max number of paths per sentence')
tf.app.flags.DEFINE_integer('max_path_len', 7, 'max number of tokens per path')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 25, 'number of train iter')
tf.app.flags.DEFINE_integer('run_times', 1, 'repeat times of this model')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('lr_main', 0.005, 'learning rate')
tf.app.flags.DEFINE_integer('main_decay_step', 60, 'main learning rate decay step')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('lr_decay', 0.5, 'learning rate decay')
tf.app.flags.DEFINE_integer('seed', 0, 'random seed')
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_float('l2_reg', 1e-2, 'l2 regularization')
# Clause Encode ##
tf.app.flags.DEFINE_integer('num_heads', 5, 'the num heads of attention')#head
tf.app.flags.DEFINE_integer('n_layers', 3, 'the layers of transformer beside main')
# GCN ##
tf.app.flags.DEFINE_integer('n_hops', 1, 'the layers of transformer beside main')#graph_layer
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'GCN layer dropout keep prob') # graph 出来以后的结果
tf.app.flags.DEFINE_integer('edge_type', 1, '2 for s-edge and k-edge, 3 for extra r-edge')


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

def kat_model(x, sen_len, doc_len, word_dis, word_embedding, adj, emotion_pos, pos_embedding, path_data_op, path_len_op, keep_prob1, keep_prob2, RNN = func.biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    word_dis = tf.nn.embedding_lookup(pos_embedding, word_dis)
    sh2 = 2 * FLAGS.n_hidden
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])
    with tf.name_scope('word_encode'):
        wordEncode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer')
    wordEncode = tf.reshape(wordEncode, [-1, FLAGS.max_sen_len, sh2])
    with tf.name_scope('attention'):
        w1 = func.get_weight_varible('word_att_w1', [sh2, sh2])
        b1 = func.get_weight_varible('word_att_b1', [sh2])
        w2 = func.get_weight_varible('word_att_w2', [sh2, 1])
        senEncode = func.att_var(wordEncode, sen_len, w1, b1, w2)
    senEncode = tf.reshape(senEncode, [-1, FLAGS.max_doc_len, sh2])
    word_dis = tf.reshape(word_dis[:, :, 0, :], [-1, FLAGS.max_doc_len, FLAGS.embedding_dim_pos])
    senEncode_dis = tf.concat([senEncode, word_dis], axis=2)
    n_feature = 2 * FLAGS.n_hidden + FLAGS.embedding_dim_pos
    out_units = 2 * FLAGS.n_hidden
    batch = tf.shape(senEncode)[0]
    if FLAGS.n_layers > 1:
        senEncode = trans_func(senEncode_dis, senEncode, n_feature, out_units, 'layer1')
    for i in range(2, FLAGS.n_layers):
        n_feature = out_units
        senEncode = trans_func(senEncode, senEncode, n_feature, out_units, 'layer' + str(i))

    # >>>>>>>>>>>>>> Context-Aware Path Representation <<<<<<<<< #
    # document rep ##
    doc_outunits = 2*out_units
    with tf.name_scope('clause_attention'):
        w1 = func.get_weight_varible('clause_att_w1', [doc_outunits, out_units])
        b1 = func.get_weight_varible('clause_att_b1', [out_units])
        w2 = func.get_weight_varible('clause_att_w2', [out_units, 1])
        doc_rep = func.emotion_attend(senEncode, doc_len, emotion_pos,w1,w2,b1)
    #path encoding ##
    with tf.name_scope('path_word_encoder'):
        path_data_op = tf.nn.embedding_lookup(word_embedding,path_data_op)
        path_inputs = tf.reshape(path_data_op, [-1, FLAGS.max_path_len, FLAGS.embedding_dim])
        sh2 = 2 * FLAGS.n_hidden
        path_inputs = tf.nn.dropout(path_inputs, keep_prob=keep_prob1)
        path_len = tf.reshape(path_len_op, [-1])
        pathWordEncode = RNN(path_inputs, path_len,n_hidden=FLAGS.n_hidden,scope='path_encode_layer')
        pathWordEncode = tf.reshape(pathWordEncode, [-1, FLAGS.max_path_len, sh2])
    with tf.name_scope('path_sen_attention'):
        w1 = func.get_weight_varible('pathword_att_w1', [sh2, sh2])
        b1 = func.get_weight_varible('pathword_att_b1', [sh2])
        w2 = func.get_weight_varible('pathword_att_w2', [sh2, 1])
        pathSenEncode = func.att_var(pathWordEncode, path_len, w1,b1,w2) #
    pathSenEncode = tf.reshape(pathSenEncode, [-1, FLAGS.max_doc_len, FLAGS.max_path_num, sh2])
    # document rep attends the path representation ##
    with tf.name_scope('doc_query_paths'):
        '''
        Q: document_rep[?, D1]
        C: pathEncode[?, X,L,D2]
        path_len_op: [?,X,L]
        '''
        Q = doc_rep 
        Q = tf.expand_dims(Q,1)
        Q = tf.tile(Q,[1,75,1])
        expand_Q = tf.expand_dims(Q,-1)
        feature = pathSenEncode
        dot_prod = tf.matmul(feature,expand_Q) #[?,X,L,1]
        dmask = tf.to_float(tf.zeros_like(path_len_op)!=path_len_op)
        dmask = tf.expand_dims(dmask,-1) #[?,X,L,1] 
        attention_weight = mask_logits(dot_prod, dmask)  # (N,X, L ,1)
        attention = tf.nn.softmax(attention_weight, dim=2)  # (N, X,L, 1)
        att_path = tf.matmul(tf.transpose(feature,[0,1,3,2]), attention)  # (N,X, D, 1)
        att_path = tf.squeeze(att_path,-1) #[N,X,D]

    '''GAT model'''
    senEncode_main = senEncode
    nodes_mask = tf.tile(tf.expand_dims(tf.range(FLAGS.max_doc_len, dtype=tf.int32), 0),
                         # list range(500)->shape(1,500)->shape(shape(self.nodes_length)[0],500)->
                         (tf.shape(doc_len)[0], 1)) < tf.expand_dims(doc_len, -1)
    nodes_mask = tf.cast(nodes_mask, tf.float32)
    for i in range(FLAGS.n_hops):
        # senEncode_main, nodeatt= attpath_GCN(senEncode_main, att_path, adj, emotion_pos, nodes_mask,doc_len)
        senEncode_main= path_GCN(senEncode_main, att_path, adj, emotion_pos, nodes_mask, str(i)+"glayer")
        senEncode_main = tf.nn.dropout(senEncode_main, keep_prob=keep_prob2)


    #concate the hidden tensors and the emotion hidden tensors
    batch_idx = tf.expand_dims(tf.range(0,tf.shape(emotion_pos)[0]),1)
    emotion_idx = tf.expand_dims(emotion_pos,1)
    emotion_idx = tf.concat([batch_idx,emotion_idx],1)
    emotion_tensor = tf.gather_nd(senEncode_main,emotion_idx)#[bs,out_units]
    emotion_tensors = tf.stack([emotion_tensor for _ in range(senEncode_main.shape[1])],1) #[?,doc_len,out_units]

    emotion_tensors = tf.nn.dropout(emotion_tensors, keep_prob=keep_prob2)
    emotional_nodes = tf.concat([senEncode_main,emotion_tensors],-1)
    '''Predict the label'''
    pred, reg = senEncode_softmax(emotional_nodes, 'softmax_w', 'softmax_b', 2*out_units, doc_len)  # senEnc
    return pred, reg, attention, att_path

def run():
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    #load data
    x_data, y_data, sen_len_data, doc_len_data, word_distance, word_embedding, pos_embedding, adj, emotion_pos_data, path_data, path_num_data, path_len_data,id2word\
        = func.load_data(edge_type=FLAGS.edge_type)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')
    print('build model...')
    start_time = time.time()
    #create placeholders
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    emotion_pos = tf.placeholder(tf.int32, [None])
    adj_tensor = tf.placeholder(tf.float32, [None, FLAGS.edge_type, FLAGS.max_doc_len, FLAGS.max_doc_len])
    path_data_op = tf.placeholder(tf.int32,[None, FLAGS.max_doc_len, FLAGS.max_path_num, FLAGS.max_path_len]) 
    path_num_op = tf.placeholder(tf.int32,[None, FLAGS.max_doc_len])
    path_len_op = tf.placeholder(tf.int32,[None, FLAGS.max_doc_len, FLAGS.max_path_num])
    placeholders = [x, y, sen_len, doc_len, word_dis, adj_tensor, emotion_pos,path_data_op, path_num_op, path_len_op, keep_prob1, keep_prob2]
    #build graph
    pred, reg, att_path, path_rep = kat_model(x, sen_len, doc_len, word_dis, word_embedding, adj_tensor,emotion_pos, pos_embedding, path_data_op,path_len_op,keep_prob1, keep_prob2)

    with tf.name_scope('loss'):
        epsilon = 1e-4
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_op = - tf.reduce_sum(y * tf.log(pred+epsilon)) / valid_num + reg * FLAGS.l2_reg

    # Modified for learning rate decay
    global_step = tf.Variable(0, name='global_step', trainable=False)
    add_global_op = global_step.assign_add(1)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_op)
    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred, 2)

    print('build model done!\n')
    prob_list_pr, y_label = [], []
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tf.set_random_seed(1234)
    test_splits_doc, test_splits_y, test_splits_pre =[],[],[]
    with tf.Session(config=tf_config) as sess:
        kf, fold, SID = KFold(n_splits=10, shuffle=True), 1, 0
        Id = []
        Id_2 = []
        p_list, r_list, f1_list = [], [], []
        parameter_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(parameter_num)
        split_num = 0
        for train, test in kf.split(x_data):
            split_num += 1
            print("Split:%d/10"%split_num)

            tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, tr_adj_data, tr_emotion_pos, tr_path, tr_path_num, tr_path_len = map(lambda x: x[train],
            [x_data, y_data, sen_len_data, doc_len_data, word_distance, adj, emotion_pos_data, path_data, path_num_data, path_len_data])

            te_x, te_y, te_sen_len, te_doc_len, te_word_dis, te_adj_data, te_emotion_pos,te_path, te_path_num, te_path_len = map(lambda x: x[test],
            [x_data, y_data, sen_len_data, doc_len_data, word_distance, adj, emotion_pos_data, path_data, path_num_data, path_len_data])

            precision_list, recall_list, FF1_list = [], [], []
            pre_list, true_list, pre_list_prob = [], [], []

            Id_2.append(test)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            sess.run(tf.global_variables_initializer()) 
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = -1
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))
            val_report_epoch = []
            train_loss_epoch = []
            valid_loss_epoch = []
            '''*********Train********'''
            for epoch in range(FLAGS.training_iter):
                step = 1
                train_report = []
                train_loss = []
                valid_loss = []
                # best_pre = []
                for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, tr_adj_data, tr_emotion_pos,tr_path, tr_path_num, tr_path_len, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                    _, loss, pred_y, true_y, pred_prob, doc_len_batch,_ = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, pred, doc_len,add_global_op],
                        feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1,= func.acc_prf(pred_y, true_y, doc_len_batch)
                    train_report.append((acc, p, r, f1))
                    train_loss.append(loss)
                    if step % 15 == 0:
                        print('epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc))
                    step = step + 1
                train_report = np.array(train_report)
                train_report_mean = np.mean(train_report, axis=0)
                train_loss_mean = np.mean(np.array(train_loss),axis=0)
                
                print(train_loss_mean.squeeze().tolist())
                train_loss_epoch.append(train_loss_mean.squeeze().tolist())

                '''*********Test********'''
                test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, te_adj_data, te_emotion_pos,te_path, te_path_num, te_path_len, 1., 1.]
                loss, pred_y, true_y, pred_prob, path_att_weights, path_rep_data= sess.run(
                    [loss_op, pred_y_op, true_y_op, pred, att_path, path_rep], feed_dict=dict(zip(placeholders, test)))
                # valid_loss.append(loss)
                end_time = time.time()

                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                acc, p, r, f1 = func.acc_prf(pred_y, true_y, te_doc_len)

                val_report_epoch.append((acc, p, r, f1))
                valid_loss_epoch.append(loss)
                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)
                if f1 > max_f1: #save the best prediction
                    best_pre = pred_y
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\ntest: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc, p, r, f1, max_f1))

            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            print("maxIndex:", maxIndex)
            print('Optimization Finished!\n')
            pred_prob = pre_list_prob[maxIndex]

            for i in range(pred_y.shape[0]):
                for j in range(te_doc_len[i]):
                    prob_list_pr.append(pred_prob[i][j][1])
                    y_label.append(true_y[i][j])

            print("*********prob_list_pr", len(prob_list_pr))
            print("*********y_label", len(y_label))

            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)

        print("running time: ", str((end_time - start_time) / 60.))
        print("running time: ", str((end_time - start_time) / 60.))
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])

        print("f1_score in 10 fold: {}\naverage : {} {} {}\n".format(np.array(f1_list).reshape(-1, 1), round(p, 4), round(r, 4), round(f1, 4)))
        # f.close()
        return p, r, f1


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, learning_rate-{}, keep_prob1-{}, num_heads-{}, n_layers-{}'.format(
        FLAGS.batch_size, FLAGS.lr_main, FLAGS.keep_prob1, FLAGS.num_heads, FLAGS.n_layers))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, y, sen_len, doc_len, word_dis, adj, emotion_pos,path, path_num, path_len,keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], adj[index],emotion_pos[index],
                     path[index], path_num[index], path_len[index],keep_prob1, keep_prob2]
        yield feed_list, len(index)

def senEncode_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg


def trans_func(senEncode_dis, senEncode, n_feature, out_units, scope_var):
    senEncode_assist = trans.multihead_attention(queries=senEncode_dis,
                                            keys=senEncode_dis,
                                            values=senEncode,
                                            units_query=n_feature,
                                            num_heads=FLAGS.num_heads,
                                            dropout_rate=0,
                                            is_training=True,
                                            scope=scope_var)
    senEncode_assist = trans.feedforward_1(senEncode_assist, n_feature, out_units)
    return senEncode_assist

    # senEncode_main, att_path, adj, emotion_pos, nodes_mask, str(i)+"glayer"
def GCNLayer2(hidden_tensor, path_data, adj, emotion_pos,hidden_mask,scope_var):
    with tf.variable_scope(scope_var):
        print(adj.shape[1])
        adjacency_tensor = adj
        hidden_tensors = tf.stack([tf.layers.dense(inputs=hidden_tensor, units=hidden_tensor.shape[-1])
                                   for _ in range(adj.shape[1])], 1) * \
                         tf.expand_dims(tf.expand_dims(hidden_mask, -1), 1)

        update = tf.reduce_sum(tf.matmul(adjacency_tensor, hidden_tensors), 1) + tf.layers.dense(
            hidden_tensor, units=hidden_tensor.shape[-1]) * tf.expand_dims(hidden_mask, -1)

        att = tf.layers.dense(tf.concat((update, hidden_tensor), -1), units=hidden_tensor.shape[-1],
                              activation=tf.nn.sigmoid) * tf.expand_dims(hidden_mask, -1)

        output = att * tf.nn.tanh(update) + (1 - att) * hidden_tensor #jingjing said tanh就不用管了
        return output

def path_GCN(hidden_tensor, path_data, adj, emotion_pos,hidden_mask,scope_var):
        adjacency_tensor = adj
        path_hidden = tf.concat([hidden_tensor,path_data],axis = -1)
        path_hidden =tf.layers.dense(path_hidden, units=hidden_tensor.shape[-1])

        # hidden_tensors = tf.stack([tf.layers.dense(inputs=hidden_tensor, units=hidden_tensor.shape[-1]),path_hidden],1)*tf.expand_dims(tf.expand_dims(hidden_mask, -1), 1)

        hidden_tensors = tf.expand_dims(hidden_tensor,1)

        update = tf.reduce_sum(tf.matmul(adjacency_tensor, hidden_tensors), 1) + tf.layers.dense(hidden_tensor, units=hidden_tensor.shape[-1]) * tf.expand_dims(hidden_mask, -1)

        att = tf.layers.dense(tf.concat((update, hidden_tensor), -1), units=hidden_tensor.shape[-1],
                              activation=tf.nn.sigmoid) * tf.expand_dims(hidden_mask, -1)

        output = att * tf.nn.tanh(update) + (1 - att) * hidden_tensor
        return output

def attpath_GCN(hidden_tensor, path_data, adj, emotion_pos,hidden_mask,doc_len):
        '''the similarity between "hidden_tensor+path" and "emotion cause" is the attention'''
        dense_path_rep = tf.layers.dense(path_data,units=30,activation='selu')
        hidden_tensor = tf.layers.dense(tf.concat([hidden_tensor,dense_path_rep],-1),units=hidden_tensor.shape[-1],activation='selu')
        hidden_tensor = tf.layers.dense(hidden_tensor,units=hidden_tensor.shape[-1],activation='selu')
        left_tensor = hidden_tensor
        batch_idx = tf.expand_dims(tf.range(0,tf.shape(emotion_pos)[0]),1)
        emotion_idx = tf.expand_dims(emotion_pos,1)
        emotion_idx = tf.concat([batch_idx,emotion_idx],1)
        emotion_tensor = tf.gather_nd(hidden_tensor,emotion_idx)#[bs,out_units]
        emotion_tensors = tf.stack([emotion_tensor for _ in range(hidden_tensor.shape[1])],1) #[?,doc_len,out_units]
        dot_att = False
        self_att = True
        clip_att = True
        if dot_att:
            ori_prod = tf.matmul(tf.layers.dense(left_tensor,units=left_tensor.shape[-1],activation='relu',name='att_key',reuse=tf.AUTO_REUSE),tf.layers.dense(emotion_tensors,units=emotion_tensors.shape[-1],activation='relu',name='att_query',reuse = tf.AUTO_REUSE),transpose_b=True) #[?,75,1]
            masked_prod = mask_logits(ori_prod,adj[:,1,:])#1-dim in adj is the k-edge
            score = tf.nn.softmax(masked_prod/tf.to_ft(hidden_tensor.shape[-1]))#[?,1,75]
            update = tf.matmul(score,tf.layers.dense(hidden_tensor,units=left_tensor.shape[-1],activation='relu',name='att_value',reuse=tf.AUTO_REUSE))+tf.matmul(adj[:,0,:,:],hidden_tensor)#add different weights on the second adj
        elif self_att: #normed head/relation and normed tail to calculate the attention score/similarity
            ex_emotion_tensor = tf.layers.dense(tf.expand_dims(emotion_tensor,1),units=hidden_tensor.shape[-1],activation='selu')
            key = ex_emotion_tensor
            value = key
            query = tf.layers.dense(path_data,units=path_data.shape[-1],activation='selu')
            # query = hidden_tensor
            score = tf.transpose(tf.matmul(query,key,transpose_b=True),(0,2,1))#[?,75,f][?,f,1]->[?,75,1]
            alpha = func.softmax_by_length(score,doc_len)
            k_nei_vector = tf.transpose(alpha,(0,2,1))*ex_emotion_tensor
            s_nei_vector = tf.matmul(adj[:,0,:,:],hidden_tensor) #[?,75,75] [?,75,dim]
            update = k_nei_vector+s_nei_vector
        att = tf.layers.dense(tf.concat((update, hidden_tensor), -1), units=hidden_tensor.shape[-1], activation=tf.nn.sigmoid) * tf.expand_dims(hidden_mask, -1)

        output = att * tf.nn.tanh(update) + (1 - att) * hidden_tensor
        return output,tf.squeeze(alpha,1)

def main(_):
    grid_search = {}
    params = {
    "lr_main":[0.001],
    "n_layers": [2],
    "n_hops":[1,2,3,4,5],
    "keep_prob1":[0.5],
    "keep_prob2":[0.8],
    "lr_decay":[0.5],
    "l2_reg":[1e-4],
    "batch_size":[64],
    "num_heads":[5],
    "seed":[2020]
    }

    params_search = list(ParameterGrid(params))  #

    for i, param in enumerate(params_search):
        print("*************seed_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(FLAGS, key, value)
        p_list, r_list, f1_list = [], [], []
        for j in range(FLAGS.run_times):
            print("*************run(){}*************".format(j + 1))
            p, r, f1 = run()
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        for j in range(FLAGS.run_times):
            print(round(p_list[j], 4), round(r_list[j], 4), round(f1_list[j], 4))
        print("avg_prf: ", np.mean(p_list), np.mean(r_list), np.mean(f1_list))

        grid_search[str(param)] = {"PRF": [round(np.mean(p_list), 4), round(np.mean(r_list), 4),
                                           round(np.mean(f1_list), 4)]}

    for key, value in grid_search.items():
        print("Main: ", key, value)


if __name__ == '__main__':
    tf.app.run()
