"""Adversarial attack on PAEDAL model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import csv
import ad_graph 
import sys, os, time, codecs, pdb
import utils
from utils.tf_funcs import *
from sklearn.model_selection import KFold
import random
import os
from utils.prepare_data import load_w2v, load_data, acc_prf, print_training_info
from sklearn.metrics import precision_score, recall_score, f1_score

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('train_dir','./data/','Directory for logs and checkpoints.')
#>>>>>>>>>>>>>>>>>>>>> For models <<<<<<<<<<<<<<<<<<<<<<#
tf.app.flags.DEFINE_string('model_type', 'peadgl', 'embedding file')
tf.app.flags.DEFINE_string('mode', 'train_adv', 'embedding file')
# >>>>>>>>>>>>>>>>>>>> For peaModel <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
tf.app.flags.DEFINE_string('pos_trainable', '', 'whether position embedding is trainable')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of sentences per documents')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_string('use_position', 'PAE', 'PAE or PEC' )
tf.app.flags.DEFINE_string('use_DGL', 'use', 'whether use DGL')
tf.app.flags.DEFINE_string('hierachy', '', 'whether use hierachy')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('train_file_path', './data/clause_keywords.csv', 'training file')
tf.app.flags.DEFINE_string('log_file_name', 'PAEDGL.log', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 20, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'PAEDGL', 'RNN scope')
tf.app.flags.DEFINE_integer('test_steps', 200, 'test at every step')
tf.app.flags.DEFINE_integer('train_steps', 20, 'show statistics of training')
tf.app.flags.DEFINE_integer("every", 1, "one sample generate 2 negative sample")  # number of batches negtive samples
# not easy to tune 
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of samples per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization rate')
tf.app.flags.DEFINE_float('lambda1', 0.1, 'rate for position prediction loss')

tf.app.flags.DEFINE_integer('dis_warm_up_step', 50, 'discriminator warm up step')
tf.app.flags.DEFINE_integer('gene_warm_up_step', 50, 'generator warm up step')

#>>>>>>>>>>>>>>>>>>For generator <<<<<<<<<<<<<<<<<<<#
tf.app.flags.DEFINE_float('generator_learning_rate', 0.001, 'rate for position prediction loss')
tf.app.flags.DEFINE_float('max_grad_norm', 1.0,'Clip the global gradient norm to this value.')

tf.app.flags.DEFINE_integer('random_seed', 29,'random_seed')
tf.app.flags.DEFINE_integer('max_steps', 100000,'random_seed')

"""
checkpoint_dir = FLAGS.train_dir,
save_checkpoint_step = FLAGS.save_step,
hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_step),
LoggerHook(dis_loss,dis_acc,FLAGS.display_step,FLAGS.batch_size,print)
EvaluateHook(eval_acc,FLAGS.eval_step,FLAGS.eval_start_step,print)
"""

def train(model=None):
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    # Model Code Block
    print("enter the run")
    word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, FLAGS.train_file_path, FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    if FLAGS.pos_trainable:
        print('pos_embedding trainable!')
        pos_embedding = tf.Variable(pos_embedding, dtype=tf.float32, name='pos_embedding')
    else:
        print("pos_embedding is constant")
        pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    print('build model...')
    
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102])
    placeholders = [x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, y, y_p, label_pos, ]
    
    
    pred_c_tr, pred_c_te, pred_p, reg = model.build_model(word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2)
    valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
    loss_cause = - tf.reduce_sum(y * tf.log(pred_c_tr)) / valid_num
    loss_position = - tf.reduce_sum(y_p * tf.log(pred_p)) / valid_num
    loss_op = loss_cause + loss_position * FLAGS.lambda1 + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    print('lambda1: {}'.format(FLAGS.lambda1))
    
    
    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred_c_tr, 2)
    pred_y_op_te = tf.argmax(pred_c_te, 2)
    print('build model done!\n')
    # Data Code Block
    y_p_data, y_data, x_data, sen_len_data, doc_len_data, word_distance, DGL_data, label_pos_data,emotion_pos = load_data(FLAGS.train_file_path, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
    # Training Code Block
    print_training_info(FLAGS)
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    with tf.Session() as sess:
    # with tf.Session() as sess:
        kf, fold = KFold(n_splits=10), 1
        p_list, r_list, f1_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_y_p, tr_sen_len, tr_doc_len, tr_word_dis, tr_DGL, tr_label_pos = map(lambda x: x[train],
                [x_data, y_data, y_p_data, sen_len_data, doc_len_data, word_distance, DGL_data, label_pos_data])
            te_x, te_y, te_y_p, te_sen_len, te_doc_len, te_word_dis, te_DGL,te_label_pos = map(lambda x: x[test],
                [x_data, y_data, y_p_data, sen_len_data, doc_len_data, word_distance, DGL_data,label_pos_data])
            
            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))
            for i in range(FLAGS.training_iter):
                start_time = time.time() 
                step = 1
                # train
                for train, _ in model.get_batch_data(tr_x, tr_word_dis, tr_DGL, tr_label_pos, emotion_pos, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, tr_y_p, FLAGS.batch_size):
                    #!!!run here
                    _, loss, loss_c, loss_p, pred_y, true_y, doc_len_batch = sess.run(
                        [optimizer, loss_op, loss_cause, loss_position, pred_y_op, true_y_op, doc_len], feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1 = acc_prf(pred_y, true_y, doc_len_batch)
                    print('step {}: loss {:.4f} loss_cause {:.4f} loss_position {:.4f} acc {:.4f} \np {:.4f} r {:.4f} f1 {:.4f}'.format(step, loss, loss_c, loss_p, acc, p, r, f1 ))
                    step = step + 1
                # test
                test = [te_x, te_word_dis, te_DGL, te_sen_len, te_doc_len, 1., 1., te_y, te_y_p]

                loss, pred_y_te, true_y, doc_len_batch = sess.run(
                        [loss_op, pred_y_op_te, true_y_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                acc, p, r, f1 = acc_prf(pred_y_te, true_y, doc_len_batch)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\nepoch {}: loss {:.4f} acc {:.4f}\np {:.4f} r {:.4f} f1 {:.4f} max_f1 {:.4f}'.format(i, loss, acc, p, r, f1, max_f1 ))
                print("cost time: {:.1f}s\n".format(time.time()-start_time))
            print('Optimization Finished!\n')
            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)  
        print_training_info(FLAGS)
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])
        print("f1_score in 10 fold: {}\naverage : p {} r {} f1 {}\n".format(np.array(f1_list).reshape(-1,1), p, r, f1))

def train_adv(model=None):
    global_step = tf.Variable(0, trainable=False)
    add_global = tf.assign_add(global_step, 1)

    word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, FLAGS.train_file_path, FLAGS.w2v_file)

    y_p_data, y_data, x_data, sen_len_data, doc_len_data, word_distance, DGL_data, label_pos_data, emotion_pos_data = load_data(FLAGS.train_file_path, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)

    train_doc = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    train_word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len])
    train_sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    train_doc_len = tf.placeholder(tf.int32, [None])
    train_keep_prob1 = tf.placeholder(tf.float32)
    train_keep_prob2 = tf.placeholder(tf.float32)
    train_label = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    train_label_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102])
    train_label_pos_op = tf.placeholder(tf.int32, [None])
    train_emotion_pos_op = tf.placeholder(tf.int32,[None])


    placeholders = [train_doc, train_word_dis, DGL, train_sen_len, train_doc_len, train_keep_prob1, train_keep_prob2, train_label, train_label_p, train_label_pos_op, train_emotion_pos_op]

    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    if FLAGS.pos_trainable:
        print('pos_embedding trainable!')
        pos_embedding = tf.Variable(pos_embedding, dtype=tf.float32, name='pos_embedding')
    else:
        print("pos_embedding is constant")
        pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    print('build model...')

    action_prob_op, gene_loss_op, train_gene_op = model.train_generator(global_step,word_embedding, pos_embedding)
    dis_loss_op, train_dis_op, reward_op, pred_y, gt = model.train_discriminator(global_step,word_embedding, pos_embedding)
    # initialize prob
    original_prob_op = model.get_original_prob(word_embedding, pos_embedding) #

    paedgl_loss_op, train_paedgl_op,paedgl_pre_op,paedgl_gt_op= model.paedgl(global_step,word_embedding, pos_embedding)

    train_ckpt_dir = FLAGS.train_dir+'/train_ckpt'
    os.makedirs(train_ckpt_dir, exist_ok=True)
    sum_writer = tf.summary.FileWriter(str(train_ckpt_dir), graph=tf.get_default_graph())
    best_dev_f1, best_attack_f1 = 0.0,0.0
    final_acc = 0.0

    average_reward = 0
    all_reward = 0
    all_sent_num = 0
    p_list, r_list, f1_list=[],[],[]
    att_plist, attr_list,attf1_list=[],[],[]
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        
        iteration = 0
        n_split = 0
        ad_documents = []
        kf, fold = KFold(n_splits=10), 1
        for train, test in kf.split(x_data):
            print("The n_split is %d"%(n_split))
            best_dev_f1 = 0.0
            best_attack_f1 = 0.0
            tr_x, tr_y, tr_y_p, tr_sen_len, tr_doc_len, tr_word_dis, tr_DGL, tr_label_pos, tr_emotion_pos = map(lambda x: x[train],
            [x_data, y_data, y_p_data, sen_len_data, doc_len_data, word_distance, DGL_data, label_pos_data, emotion_pos_data])

            te_x, te_y, te_y_p, te_sen_len, te_doc_len, te_word_dis, te_DGL, te_label_pos,te_emotion_pos = map(lambda x: x[test],
            [x_data, y_data, y_p_data, sen_len_data, doc_len_data, word_distance, DGL_data,label_pos_data,emotion_pos_data])

            test_data = [te_x, te_word_dis, te_DGL, te_sen_len, te_doc_len, 1., 1., te_y, te_y_p, te_label_pos,te_emotion_pos]
            sess.run(tf.global_variables_initializer())
            this_global_step = 0
            print("*******************************")
            # print("RUN THE n_split:%d:"%(n_split))

            for _ in range(FLAGS.training_iter): #15
                
                iteration = iteration +1
                
                for train,_ in model.get_batch_data(tr_x, tr_word_dis, tr_DGL, tr_label_pos, tr_emotion_pos, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, tr_y_p, FLAGS.batch_size): #once dataset
                    '''Step1: generate train_data for discriminator'''
                    train_doc_data,train_sen_len_data,train_label_data,train_word_dis_data,train_DGL_data,train_doc_len_data,train_label_p_data, train_label_pos_data, train_emotion_pos_data = sess.run([train_doc,train_sen_len,train_label,train_word_dis,DGL,train_doc_len,train_label_p, train_label_pos_op, train_emotion_pos_op], 
                    feed_dict=dict(zip(placeholders, train)))
                    this_global_step = sess.run(add_global)
                    raw_sentence = train_doc_data.copy()
                    '''Step2: warm up the discriminator'''
                    if this_global_step < FLAGS.dis_warm_up_step:
                        dis_loss, _,= sess.run([dis_loss_op, train_dis_op],feed_dict={
                            'discriminator/train_doc:0': train_doc_data,
                            'discriminator/train_sen_len:0': train_sen_len_data,
                            'discriminator/train_label:0': train_label_data,
                            'discriminator/train_word_dis:0':train_word_dis_data,
                            'discriminator/train_DGL:0':train_DGL_data,
                            'discriminator/train_doc_len:0':train_doc_len_data,
                            'discriminator/train_p_label:0':train_label_p_data,
                            'discriminator/keep_prob1:0':FLAGS.keep_prob1,
                            'discriminator/keep_prob2:0':FLAGS.keep_prob2,
                        })
                        '''Step3: warm up generator'''
                    elif this_global_step < FLAGS.gene_warm_up_step+FLAGS.dis_warm_up_step:
                        original_prob_data = sess.run(original_prob_op, feed_dict={
                            'original/ori_x:0':train_doc_data,
                            'original/ori_sen_len:0':train_sen_len_data,
                            'original/ori_label:0':train_label_data,
                            'original/ori_word_dis:0':train_word_dis_data,
                            'original/ori_DGL:0':train_DGL_data,
                            'original/ori_doc_len:0':train_doc_len_data,
                            'original/ori_keep_prob1:0':FLAGS.keep_prob1,
                            'original/ori_keep_prob2:0':FLAGS.keep_prob2,
                            'original/ori_p_label:0':train_label_p_data

                        })
                        action = sess.run(action_prob_op, feed_dict={
                            'generator/train_doc:0': train_doc_data,
                            'generator/train_sen_len:0': train_sen_len_data,
                            'generator/train_label:0': train_label_data,
                            'generator/train_word_dis:0':train_word_dis_data,
                            'generator/train_DGL:0':train_DGL_data,
                            'generator/train_doc_len:0':train_doc_len_data,
                            'generator/train_p_label:0':train_label_p_data,
                            'generator/keep_prob1:0':FLAGS.keep_prob1,
                            'generator/keep_prob2:0':FLAGS.keep_prob2,
                            'generator/emotion_pos:0':train_emotion_pos_data
                                    }
                        )                       
                        doc_new_data, action_idx, c1,c2,train_new_sen_len_data, train_new_label_data, train_new_dgl_data = model.generate_new_sentence_with_action(
                                                            action,
                                                            train_doc_data,
                                                            train_doc_len_data,
                                                            train_sen_len_data,
                                                            train_label_data,
                                                            train_DGL_data,
                                                            train_emotion_pos_data)
                        reward = sess.run(reward_op, feed_dict={
                            'discriminator/train_doc:0': doc_new_data,
                            'discriminator/original_prob:0':original_prob_data,
                            'discriminator/train_sen_len:0': train_new_sen_len_data,
                            'discriminator/train_label:0': train_new_label_data,
                            'discriminator/train_word_dis:0':train_word_dis_data,
                            'discriminator/train_DGL:0':train_new_dgl_data,
                            'discriminator/train_doc_len:0':train_doc_len_data,
                            'discriminator/train_p_label:0':train_label_p_data,
                            'discriminator/keep_prob1:0':FLAGS.keep_prob1,
                            'discriminator/keep_prob2:0':FLAGS.keep_prob2,
                        })
                        all_sent_num += len(reward)
                        all_reward += np.sum(reward)
                        average_reward = all_reward/all_sent_num
                        reward -= average_reward

                        gene_loss, _=sess.run([gene_loss_op, train_gene_op], feed_dict={
                            'generator/train_doc:0': train_doc_data,
                            'generator/train_sen_len:0': train_sen_len_data,
                            'generator/train_label:0': train_label_data,
                            'generator/train_word_dis:0':train_word_dis_data,
                            'generator/train_DGL:0':train_DGL_data,
                            'generator/train_doc_len:0':train_doc_len_data,
                            'generator/train_p_label:0':train_label_p_data,
                            'generator/keep_prob1:0':FLAGS.keep_prob1,
                            'generator/keep_prob2:0':FLAGS.keep_prob2,
                            'generator/reward_score:0': reward,
                            'generator/emotion_pos:0':train_emotion_pos_data,
                            'generator/action_idx:0': action_idx})
                        dis_loss = 0
                    else:  # adversarial train
                        rand_num = random.choice([1]*FLAGS.every+[0])
                        if rand_num!=0:
                            original_prob_data = sess.run(original_prob_op, feed_dict={
                                'original/ori_x:0':train_doc_data,
                                'original/ori_sen_len:0':train_sen_len_data,
                                'original/ori_label:0':train_label_data,
                                'original/ori_word_dis:0':train_word_dis_data,
                                'original/ori_DGL:0':train_DGL_data,
                                'original/ori_doc_len:0':train_doc_len_data,
                                'original/ori_keep_prob1:0':FLAGS.keep_prob1,
                                'original/ori_keep_prob2:0':FLAGS.keep_prob2,
                                'original/ori_p_label:0':train_label_p_data

                        })
                            action = sess.run(action_prob_op, feed_dict={
                                'generator/train_doc:0': train_doc_data,
                                'generator/train_sen_len:0': train_sen_len_data,
                                'generator/train_label:0': train_label_data,
                                'generator/train_word_dis:0':train_word_dis_data,
                                'generator/train_DGL:0':train_DGL_data,
                                'generator/train_doc_len:0':train_doc_len_data,
                                'generator/train_p_label:0':train_label_p_data,
                                'generator/keep_prob1:0':FLAGS.keep_prob1,
                                'generator/keep_prob2:0':FLAGS.keep_prob2,
                                'generator/emotion_pos:0':train_emotion_pos_data
                                    }
                        )
                            doc_new_data, action_idx, c1,c2,train_new_sen_len_data, train_new_label_data, train_new_dgl_data = model.generate_new_sentence_with_action(
                                                            action,
                                                            train_doc_data,
                                                            train_doc_len_data,
                                                            train_sen_len_data,
                                                            train_label_data,
                                                            train_DGL_data,
                                                            train_emotion_pos_data)
                            dis_loss, _, reward = sess.run([dis_loss_op, train_dis_op, reward_op],feed_dict={
                            'discriminator/train_doc:0': doc_new_data,
                            'discriminator/original_prob:0':original_prob_data,
                            'discriminator/train_sen_len:0': train_new_sen_len_data,
                            'discriminator/train_label:0': train_new_label_data,
                            'discriminator/train_word_dis:0':train_word_dis_data,
                            'discriminator/train_DGL:0':train_new_dgl_data,
                            'discriminator/train_doc_len:0':train_doc_len_data,
                            'discriminator/train_p_label:0':train_label_p_data,
                            'discriminator/keep_prob1:0':FLAGS.keep_prob1,
                            'discriminator/keep_prob2:0':FLAGS.keep_prob2,
                        })
                            all_sent_num += len(reward)
                            all_reward += np.sum(reward)
                            average_reward = all_reward/all_sent_num
                            reward -= average_reward
                            gene_loss, _=sess.run([gene_loss_op, train_gene_op], feed_dict={
                                'generator/train_doc:0': train_doc_data,
                                'generator/train_sen_len:0': train_sen_len_data,
                                'generator/train_label:0': train_label_data,
                                'generator/train_word_dis:0':train_word_dis_data,
                                'generator/train_DGL:0':train_DGL_data,
                                'generator/train_doc_len:0':train_doc_len_data,
                                'generator/train_p_label:0':train_label_p_data,
                                'generator/keep_prob1:0':FLAGS.keep_prob1,
                                'generator/keep_prob2:0':FLAGS.keep_prob2,
                                'generator/reward_score:0': reward,
                                'generator/action_idx:0': action_idx,
                                'generator/emotion_pos:0':train_emotion_pos_data
                                })
                        else: 
                            dis_loss, _,train_pre, train_gt= sess.run([dis_loss_op, train_dis_op,pred_y,gt],feed_dict={
                            'discriminator/train_doc:0': train_doc_data,
                            'discriminator/original_prob:0':original_prob_data,
                            'discriminator/train_sen_len:0': train_sen_len_data,
                            'discriminator/train_label:0': train_label_data,
                            'discriminator/train_word_dis:0':train_word_dis_data,
                            'discriminator/train_DGL:0':train_DGL_data,
                            'discriminator/train_doc_len:0':train_doc_len_data,
                            'discriminator/train_p_label:0':train_label_p_data,
                            'discriminator/keep_prob1:0':FLAGS.keep_prob1,
                            'discriminator/keep_prob2:0':FLAGS.keep_prob2,
                        })
                            acc, p, r, f1= acc_prf(train_pre,train_gt, train_doc_len_data)
                            if this_global_step%FLAGS.train_steps==0:
                                print("****train results*********")
                                print("step:%d,ad_train_loss:%f,acc:%f,p:%f,r:%f,f1:%f"%(this_global_step,dis_loss,acc,p,r,f1))
                            paedgl_loss, _,paedgl_pre, paeddl_gt= sess.run([paedgl_loss_op, train_paedgl_op,paedgl_pre_op,paedgl_gt_op],feed_dict={
                            'paedgl/train_doc:0': train_doc_data,
                            'paedgl/original_prob:0':original_prob_data,
                            'paedgl/train_sen_len:0': train_sen_len_data,
                            'paedgl/train_label:0': train_label_data,
                            'paedgl/train_word_dis:0':train_word_dis_data,
                            'paedgl/train_doc_len:0':train_doc_len_data,
                            'paedgl/train_p_label:0':train_label_p_data,
                            'paedgl/keep_prob1:0':FLAGS.keep_prob1,
                            'paedgl/keep_prob2:0':FLAGS.keep_prob2,
                        })
                            if this_global_step%FLAGS.train_steps==0:
                                acc, p, r, f1= acc_prf(paedgl_pre,paeddl_gt, train_doc_len_data)
                                print("step:%d,paedgl_train_loss:%f,acc:%f,p:%f,r:%f,f1:%f"%(this_global_step,paedgl_loss,acc,p,r,f1))
                

                if this_global_step!=0 and  this_global_step > FLAGS.dis_warm_up_step + FLAGS.gene_warm_up_step:
                    test_doc_data,test_sen_len_data,test_label_data,test_word_dis_data,test_DGL_data,test_doc_len_data,test_label_p_data, test_emotion_pos_data\
                     = sess.run([train_doc,train_sen_len,train_label,train_word_dis,DGL,train_doc_len,train_label_p,train_emotion_pos_op],
                    
            feed_dict=dict(zip(placeholders, test_data)))
                    dev_loss,dev_pre,dev_gt = sess.run([paedgl_loss_op,paedgl_pre_op,paedgl_gt_op],feed_dict={
                        'paedgl/train_doc:0': test_doc_data,
                        'paedgl/train_sen_len:0': test_sen_len_data,
                        'paedgl/train_label:0': test_label_data,
                        'paedgl/train_word_dis:0':test_word_dis_data,
                        'paedgl/train_doc_len:0':test_doc_len_data,
                        'paedgl/train_p_label:0':test_label_p_data,
                        'paedgl/keep_prob1:0':1.0,
                        'paedgl/keep_prob2:0':1.0
                })  
                    print("")
                    print("*********test results*******")
                    acc, p, r, f1= acc_prf(dev_pre,dev_gt, test_doc_len_data)
                    print("peadgl_loss=%f,paedgl acc=%f,,p:%f,r:%f,f1:%f At global step: %d."%(dev_loss,acc,p,r,f1, this_global_step))
                    # print("")
                    if f1 > best_dev_f1: #if better dev results, save the weights
                        best_dev_f1= f1
                        best_dev_p, best_dev_r, best_dev_f1 = p, r, f1
                        print("!!!!best f1=%f n_split:%d."%(best_dev_f1,n_split))
                        print("")
                        save_checkpoint(saver, sess, FLAGS.train_dir, n_split, str(round(float(best_dev_f1),2)))

                    '''test on the adversarial samples
                    1) based on test data, generate action ->run generator
                    2) generate new data for ad_attack'''
                    #step1:
                    attack_action = sess.run(action_prob_op, feed_dict={
                    'generator/train_doc:0': test_doc_data,
                    'generator/train_sen_len:0': test_sen_len_data,
                    'generator/train_label:0': test_label_data,
                    'generator/train_word_dis:0':test_word_dis_data,
                    'generator/train_DGL:0':test_DGL_data,
                    'generator/train_doc_len:0':test_doc_len_data,
                    'generator/train_p_label:0':test_label_p_data,
                    'generator/keep_prob1:0':1.0,
                    'generator/keep_prob2:0':1.0,
                    'generator/emotion_pos:0':test_emotion_pos_data
                    })
                    #step2:
                    attack_doc_data, action_idx, c1, c2,test_attack_sen_len_data, test_attack_label_data,new_dgl_data= model.generate_new_sentence_with_action(
                                                            attack_action,
                                                            test_doc_data,
                                                            test_doc_len_data,
                                                            test_sen_len_data,
                                                            test_label_data,
                                                            test_DGL_data,
                                                            test_emotion_pos_data)
                    #step3:
                    dev_loss,dev_pre,dev_gt = sess.run([paedgl_loss_op,paedgl_pre_op,paedgl_gt_op],feed_dict={
                        'paedgl/train_doc:0': attack_doc_data, #need to place the real dev dataset
                        'paedgl/train_sen_len:0': test_attack_sen_len_data,
                        'paedgl/train_label:0': test_attack_label_data,
                        'paedgl/train_word_dis:0':test_word_dis_data,
                        'paedgl/train_doc_len:0':test_doc_len_data,
                        'paedgl/train_p_label:0':test_label_p_data,
                        'paedgl/keep_prob1:0':1.0,
                        'paedgl/keep_prob2:0':1.0
                })

                    acc, p, r, f1= acc_prf(dev_pre,dev_gt, test_doc_len_data)
                    print("ATTCK_paedgl_loss=%f,dev acc=%f,,p:%f,r:%f,f1:%f At global step: %d."%(dev_loss,acc,p,r,f1, this_global_step))
                    if f1 > best_attack_f1:
                        best_attack_f1= f1
                        best_attack_p, best_attack_r, best_attack_f1 = p, r, f1
                        print("!!!!best attack f1=%f n_split:%d."%(best_attack_f1,n_split))
            att_plist.append(best_attack_p)
            attr_list.append(best_attack_r)
            attf1_list.append(best_attack_f1)

            p_list.append(best_dev_p)
            r_list.append(best_dev_r)
            f1_list.append(best_dev_f1)
            n_split = n_split +1


        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])
        print("original_model: f1_score in 10 fold: {}\naverage : p {} r {} f1 {}\n".format(np.array(f1_list).reshape(-1,1), p, r, f1))

        p, r, f1 = map(lambda x: np.array(x).mean(), [att_plist, attr_list, attf1_list])
        print("attacked_model:f1_score in 10 fold: {}\naverage : p {} r {} f1 {}\n".format(np.array(attf1_list).reshape(-1,1), p, r, f1))

def save_checkpoint(saver=None, sess=None, dir=None, n_split=None, best_dev_f1=None):
    path = os.path.join(dir,'train_ckpt','model_'+str(n_split)+'_'+best_dev_f1+'.ckpt')
    saver.save(sess, path)

def acc_prf(pred_y, true_y, doc_len):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, p, r, f1

def generate_ad_doc(doc_new_data, doc_len, sen_len, word2id, n_split, step):
    id2word={}
    csv_file =open(str(n_split)+"_"+'ad_doc.txt', 'a', newline='')
    # csvwriter = csv.writer(csv_file)
    for key in word2id.keys():
        id2word[word2id[key]]=key
    #traverse the numberic doc data
    doc_text = []
    print("iter:%d, generate data"%(n_split))
    for batch_id in range(len(doc_new_data)):
        for doc_id in range(doc_new_data[batch_id].shape[0]):
            for sen_id in range(doc_len[doc_id]):
                # print(sen_len[doc_id,sen_id])
                # print(doc_new_data[batch_id][doc_id][sen_id])
                sent = [id2word[doc_new_data[batch_id][doc_id,sen_id][word_idx]] for word_idx in range(sen_len[doc_id, sen_id])]
                # print(sent)
                sent = ''.join(sent)
                csv_file.write(sent+'\n')
    csv_file.close()
                

if __name__=='__main__':

    if FLAGS.model_type=='peadgl':
        print('loading peadgl model')
        if FLAGS.mode == 'train':
            print('train baseline ...')
            train(ad_graph.peaModel())
        if FLAGS.mode == 'train_adv': 
            print('train adv...')
            train_adv(ad_graph.peaModel())
    else:
        print('model type must be cnn or rnn')

