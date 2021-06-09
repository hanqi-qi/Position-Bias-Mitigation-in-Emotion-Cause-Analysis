
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

# Dependency imports

import tensorflow as tf
from utils.tf_funcs import *
import layers as layers_lib
# from get_data import get_dataset
# import adversarial_attacking
flags = tf.app.flags
FLAGS = flags.FLAGS

def paedgl_optimize(loss,global_step=None):
    return layers_lib.adam_optimize(loss,global_step,FLAGS.learning_rate,FLAGS.max_grad_norm)

def adam_optimize(loss,global_step=None):
    return layers_lib.adam_optimize(loss,global_step,FLAGS.learning_rate,FLAGS.max_grad_norm)

def gene_adam_optimize(loss,global_step=None):
    return layers_lib.adam_optimize(loss,global_step,FLAGS.generator_learning_rate,FLAGS.max_grad_norm)
    
def optimize(loss, global_step=None):
    return layers_lib.optimize(
        loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor)

def one2many_attention(emotion_pos, clause_reps):
    ''''
    query: emotion_clause [?,f]
    value/key: candidate_clause_reps [?,doc_len,f]
    '''
    batch_idx = tf.expand_dims(tf.range(0,tf.shape(emotion_pos)[0]),1)
    emotion_idx = tf.expand_dims(emotion_pos,1)
    emotion_idx = tf.concat([batch_idx,emotion_idx],1)
    emotion_tensor = tf.expand_dims(tf.gather_nd(clause_reps,emotion_idx),1)#[bs,1,out_units]
    emotion_pos_matrix = tf.stack([emotion_pos for _ in range(75)],1)

    scores = tf.squeeze(tf.matmul(emotion_tensor, clause_reps, transpose_b=True),1) #[?,75]
    pos_vec = tf.range(0,FLAGS.max_doc_len) #[?]
    attention = tf.nn.softmax(scores, dim=2)
    return attention

class peaModel(object):
    def __init__(self,cl_logits_input_dim=None):
        self.layers = {}
        # self.denselayer = tf.layers.Dense(units=FLAGS.max_doc_len,activation="relu")
    
    def get_doc_data(self,y_p_data, y_data, x_data, sen_len_data, doc_len_data, word_distance, DGL_data, test=False):
        for index in batch_index(len(y_data), FLAGS.batch_size, test): #convey the real value
            feed_list = [x_data[index],sen_len_data[index],doc_len_data[index], y_data[index], y_p_data[index]]
            yield feed_list#, doc_len_data[index], y_data[index], y_p_data[index]

    def logits_extractor(self,word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2,RNN, scope_name):
        with tf.variable_scope(scope_name) as scope: #TODO change the name_scope to variable_scope
            x = tf.nn.embedding_lookup(word_embedding, x)
            inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
            word_dis = tf.nn.embedding_lookup(pos_embedding, word_dis)
            sen_dis = word_dis[:,:,0,:]
            word_dis = tf.reshape(word_dis, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim_pos])
            if FLAGS.use_position == 'PAE':
                print('using PAE')
                inputs = tf.concat([inputs, word_dis], axis=2)
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
            sen_len = tf.reshape(sen_len, [-1])
            with tf.name_scope('word_encode'):  #maybe
                inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden,scope="word_layer"+scope_name)
            # inputs shape:        [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden]
            with tf.name_scope('word_attention'):
                sh2 = 2 * FLAGS.n_hidden
                w1 = get_weight_varible('word_att_w1'+scope_name, [sh2, sh2])
                b1 = get_weight_varible('word_att_b1'+scope_name, [sh2])
                w2 = get_weight_varible('word_att_w2'+scope_name, [sh2, 1])
                s = att_var(inputs,sen_len,w1,b1,w2)
            s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
            n_feature = 2 * FLAGS.n_hidden
            if FLAGS.use_position == 'PEC':
                print('using PEC')
                s = tf.concat([s, sen_dis], axis=2)
                n_feature = 2 * FLAGS.n_hidden + FLAGS.embedding_dim_pos
            # s shape:        [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden]
            if FLAGS.hierachy:
                print('use hierachy')
                s = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_layer'+scope)
                n_feature = 2 * FLAGS.n_hidden

            with tf.name_scope('softmax'):
                s = tf.reshape(s, [-1, n_feature])
                s = tf.nn.dropout(s, keep_prob=keep_prob2)
                # postion prediction
                w_p = get_weight_varible('position_w'+scope_name, [n_feature, 102])
                b_p = get_weight_varible('position_b'+scope_name, [102])
                pred_p = tf.matmul(s, w_p) + b_p
                pred_p = tf.nn.softmax(pred_p) 
                pred_p = tf.reshape(pred_p, [-1, FLAGS.max_doc_len, 102])
                # emotion cause prediction
                if FLAGS.use_DGL and scope_name!="paedgl":
                    # emotion cause prediction in training phase
                    print('using DGL feature!')
                    DGL = tf.reshape(DGL, [-1, FLAGS.max_doc_len])
                    s_tr = tf.concat([s, DGL], axis=1)
                    w = get_weight_varible('cause_w'+scope_name, [n_feature + FLAGS.max_doc_len, FLAGS.n_class])
                    b = get_weight_varible('cause_b'+scope_name, [FLAGS.n_class])
                    pred_c_tr = tf.matmul(s_tr, w) + b
                    pred_c_tr = tf.nn.softmax(pred_c_tr) 
                    #TODO copy this tensor
                    pred_c_tr = tf.reshape(pred_c_tr, [-1, FLAGS.max_doc_len, FLAGS.n_class])#TODO important for the following process
                else:
                    w = get_weight_varible('cause_w'+scope_name, [n_feature, FLAGS.n_class])
                    b = get_weight_varible('cause_b'+scope_name, [FLAGS.n_class])
                    pred_c_tr = tf.matmul(s, w) + b
                    pred_c_tr = tf.nn.softmax(pred_c_tr) 
                    pred_c_tr = tf.reshape(pred_c_tr, [-1, FLAGS.max_doc_len, FLAGS.n_class])
            
                    pred_c_te = pred_c_tr
        return s,pred_c_tr,w,b,w_p,b_p,pred_p

        
    def get_batch_data(self,x, word_dis, DGL, label_pos, emotion_pos, sen_len, doc_len, keep_prob1, keep_prob2, y, y_p, batch_size, test=False):
        for index in batch_index(y.shape[0], FLAGS.batch_size, test=False):
            feed_list = [x[index], word_dis[index], DGL[index],sen_len[index], doc_len[index], keep_prob1, keep_prob2, y[index], y_p[index],label_pos[index], emotion_pos[index]]
            yield feed_list, len(index)
    
    def get_original_prob(self,word_embedding, pos_embedding):
        with tf.variable_scope('original') as scope:
            x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name='ori_x')
            word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name='ori_word_dis')
            DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len],name='ori_DGL')
            sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len],name='ori_sen_len')
            doc_len = tf.placeholder(tf.int32, [None],name='ori_doc_len')
            keep_prob1 = tf.placeholder(tf.float32,name="ori_keep_prob1")
            keep_prob2 = tf.placeholder(tf.float32,name="ori_keep_prob2")
            y = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.n_class],name='ori_label')
            y_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102],name='ori_p_label')

            s_tr,pred_c_tr,w,b,w_p,b_p,pred_p = self.logits_extractor(word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2,RNN = biLSTM,scope_name="original")
            logits = pred_c_tr
            softmax_prob = tf.reshape(logits,[-1,logits.shape[-1]])
            new_y = tf.to_int32(tf.argmax(y,axis=2))
            original_prob = tf.gather_nd(softmax_prob, tf.stack((tf.range(tf.shape(softmax_prob)[0],dtype=tf.int32),tf.reshape(new_y,[-1])),axis=1))
            #[None]?
        return original_prob#

    def paedgl(self,global_step,word_embedding, pos_embedding):
        with tf.variable_scope('paedgl') as scope:
            #feed dict
            self.global_step = global_step
            train_doc = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_doc_len, FLAGS.max_sen_len],name="train_doc") #
            word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name='train_word_dis')
            DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len],name='train_DGL')
            sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len],name='train_sen_len')
            doc_len = tf.placeholder(tf.int32, [None],name='train_doc_len')
            keep_prob1 = tf.placeholder(tf.float32,name="keep_prob1")
            keep_prob2 = tf.placeholder(tf.float32,name="keep_prob2")
            y = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.n_class],name='train_label')
            y_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102],name='train_p_label')

            original_prob = tf.placeholder(dtype=tf.float32,shape=[None],name='original_prob')


            '''for feature extraction code'''
            s_tr,pred_c_tr,w,b,w_p,b_p,pred_p = self.logits_extractor(word_embedding, pos_embedding, train_doc, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, RNN = biLSTM, scope_name="paedgl")

            logits = pred_c_tr #pre_c_tr is a (1)tensor after softmax. (2)shape is important![-1,max_doc_len,n_class], 
            dis_softmax_prob = tf.reshape(logits,[-1,logits.shape[-1]]) #[-1,n_class])

            reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b) + tf.nn.l2_loss(w_p) + tf.nn.l2_loss(b_p)
            valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
            loss_cause = - tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(pred_c_tr)) / valid_num
            loss_position = - tf.reduce_sum(y_p * tf.log(pred_p)) / valid_num
            dis_loss = loss_cause + loss_position * FLAGS.lambda1 + reg * FLAGS.l2_reg

            # train_dis_op = paedgl_optimize(dis_loss, self.global_step)
            train_dis_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(dis_loss)

            tf.summary.scalar('dis_loss', dis_loss)

            new_y = tf.to_int32(tf.argmax(y,axis=2))
            prob = tf.gather_nd(dis_softmax_prob, tf.stack((tf.range(tf.shape(dis_softmax_prob)[0],dtype=tf.int32),tf.reshape(new_y,[-1])),axis=1))
    
            
            #for accuaracy calculation
            # true_y_op = tf.argmax(y, 2)
            pred_y_op = tf.argmax(pred_c_tr, 2)
            # pred_y_op_te = tf.argmax(pred_c_te, 2)
            prediction = tf.equal(tf.cast(pred_y_op, tf.int32),new_y)
            # acc_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

            # paedgl_loss_op, train_paedgl_op,paedgl_pre,paedgl_gt

        return dis_loss,train_dis_op,pred_y_op,new_y

    #
    def train_discriminator(self,global_step,word_embedding, pos_embedding):
        with tf.variable_scope('discriminator') as scope:
            #feed dict
            self.global_step = global_step
            train_doc = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_doc_len, FLAGS.max_sen_len],name="train_doc") #
            word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name='train_word_dis')
            DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len],name='train_DGL')
            sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len],name='train_sen_len')
            doc_len = tf.placeholder(tf.int32, [None],name='train_doc_len')
            keep_prob1 = tf.placeholder(tf.float32,name="keep_prob1")
            keep_prob2 = tf.placeholder(tf.float32,name="keep_prob2")
            y = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.n_class],name='train_label')
            y_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102],name='train_p_label')

            original_prob = tf.placeholder(dtype=tf.float32,shape=[None],name='original_prob')


            '''for feature extraction code'''
            s_tr,pred_c_tr,w,b,w_p,b_p,pred_p = self.logits_extractor(word_embedding, pos_embedding, train_doc, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, RNN = biLSTM, scope_name="dis")

            logits = pred_c_tr #pre_c_tr is a (1)tensor after softmax. (2)shape is important![-1,max_doc_len,n_class], 
            dis_softmax_prob = tf.reshape(logits,[-1,logits.shape[-1]]) #[-1,n_class])

            reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b) + tf.nn.l2_loss(w_p) + tf.nn.l2_loss(b_p)
            valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
            loss_cause = - tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(pred_c_tr)) / valid_num
            loss_position = - tf.reduce_sum(y_p * tf.log(pred_p)) / valid_num
            dis_loss = loss_cause + loss_position * FLAGS.lambda1 + reg * FLAGS.l2_reg

            # train_dis_op = adam_optimize(dis_loss, self.global_step)
            train_dis_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(dis_loss)
            

            tf.summary.scalar('dis_loss', dis_loss)

            new_y = tf.to_int32(tf.argmax(y,axis=2))
            prob = tf.gather_nd(dis_softmax_prob, tf.stack((tf.range(tf.shape(dis_softmax_prob)[0],dtype=tf.int32),tf.reshape(new_y,[-1])),axis=1))
            reward = original_prob - prob

            #for accuaracy calculation
            # true_y_op = tf.argmax(y, 2)
            pred_y_op = tf.argmax(pred_c_tr, 2)
            # pred_y_op_te = tf.argmax(pred_c_te, 2)
            prediction = tf.equal(tf.cast(pred_y_op, tf.int32),new_y)
            # acc_op = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return dis_loss,train_dis_op,reward,pred_y_op,new_y
    
    def get_generator_data(self):
        return self.train_sentence,self.train_sentence_len,self.train_label

    def train_generator(self,global_step,word_embedding, pos_embedding):
        with tf.variable_scope('generator'):
            # self.initialize_train_dataset()
            self.global_step = global_step

            train_doc = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_doc_len, FLAGS.max_sen_len],name="train_doc") #
            word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len],name='train_word_dis')
            DGL = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len],name='train_DGL')
            sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len],name='train_sen_len')
            doc_len = tf.placeholder(tf.int32, [None],name='train_doc_len')
            keep_prob1 = tf.placeholder(tf.float32,name="keep_prob1")
            keep_prob2 = tf.placeholder(tf.float32,name="keep_prob2")
            y = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.n_class],name='train_label')
            y_p = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102],name='train_p_label')
            emotion_pos = tf.placeholder(tf.int32,[None],name="emotion_pos")


            original_prob = tf.placeholder(dtype=tf.float32,shape=[None],name='original_prob')

            action_idx = tf.placeholder(dtype=tf.int32,shape=[None],name='action_idx')  #the 

            '''for feature extraction code'''
            s_tr,pred_c_tr,w,b,w_p,b_p,pred_p = self.logits_extractor(word_embedding, pos_embedding, train_doc, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, RNN = biLSTM, scope_name="gen") 
            #chooose from the following tensors as action generation-based feature, mostly according to their shape
            #s_tr[?,275] -> [-1,max_doc_len, 275]
            #pred_c_tr [-1, FLAGS.max_doc_len, FLAGS.n_class]

            action_logits = one2many_attention(emotion_pos, tf.reshape(s_tr,[-1,FLAGS.max_doc_len,s_tr.shape[-1]])) #[?,doc_len]

            reward_score = tf.placeholder(dtype=tf.float32, shape=[None],name="reward_score")

            # action_matrix = tf.squeeze(tf.layers.dense(tf.reshape(s_tr,[-1,FLAGS.max_doc_len,s_tr.shape[-1]]),1),-1)
            # action_logits = tf.nn.softmax(action_matrix,-1)
            # transform the feature to logits [None,FLAGS.max_doc_len,1]
            #transform the logits to re_sent idx
            # action_label = tf.to_float(tf.expand_dims(tf.argmax(action_logits,-1),-1))

            ori_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= action_idx,logits=action_logits)

            reward_score = tf.nn.relu(reward_score) # we don't use negative reward for training
            gene_loss = tf.reduce_mean(reward_score * tf.reduce_sum(ori_loss,-1)) 
            # train_gene_op = adam_optimize(gene_loss,self.global_step)
            train_gene_op = gene_adam_optimize(gene_loss,self.global_step)
            
        
        return action_logits,gene_loss,train_gene_op  
    
    
    def generate_new_sentence_with_action(self,
                            action=None,
                            doc=None,
                            doc_len=None,
                            sen_len=None,
                            label=None,
                            dgl_data=None,
                            emotion_pos=None):

        action_idx = []
        dis_array = 1-action
        c1_relative_list, c2_relative_list =[],[]
        for idx in range(action.shape[0]):
            clause_p1_relative = np.floor(np.random.normal(loc=-1,scale=0.5445,size=1))[0]#to a int
            c1_relative_list.append(clause_p1_relative)
            #transform the relative position to the absolute position in the sentence
            clause_p1_real = (emotion_pos[idx] + clause_p1_relative).astype(int)
            if clause_p1_real<0:
                clause_p1_real = max(0,clause_p1_real)
            elif clause_p1_real>doc_len[idx]:
                clause_p1_real = min(doc_len[idx]-1,clause_p1_real)

            clip_prob =  dis_array[idx][:doc_len[idx]]/np.sum(dis_array[idx][:doc_len[idx]])
            clause_p2 = np.random.choice(list(range(doc_len[idx])), 1, replace=True, p=clip_prob)[0]
            action_idx.append(clause_p2)
            clause_p2_relative = (clause_p2-emotion_pos[idx]).astype(int)
            c2_relative_list.append(clause_p2_relative)

            #exchange the original data
            doc[idx,[clause_p1_real,clause_p2],:] = doc[idx,[clause_p2,clause_p1_real],:]
            sen_len[idx,[clause_p1_real,clause_p2]]=sen_len[idx,[clause_p2,clause_p1_real]]
            label[idx,[clause_p1_real,clause_p2]]=label[idx,[clause_p2,clause_p1_real]]

        return doc,action_idx,c1_relative_list,c2_relative_list,sen_len,label,dgl_data

    def generate_new_sentence_with_action_random_two(self,
                            action=None,
                            doc=None,
                            doc_len=None):
        
        #action [?,doc_len]
        #doc[?,max_doc_len,sen_len]
        re_sents = []
        word_action = []
        #exchange the ex_sent with the emotion sentence
        for idx in range(action.shape[0]):
            # ex_sent = np.where(action[idx] == np.amax(action[idx]))[0][:doc_len[idx]][-1]
            #can sampling from the action[idx] probability vector like the LexicalAT
            new_prob =  action[idx][:doc_len[idx]]/np.sum(action[idx][:doc_len[idx]])
            ex_sent1,ex_sent2 = np.random.choice(list(range(len(action[idx][:doc_len[idx]]))), 2, replace=True, p=new_prob)[0]
            doc[:,[ex_sent1,ex_sent2]] = doc[:,[ex_sent1,ex_sent2]] #swap the doc
            #!!!! you should also swap the doc_len and the sen_Len
            
            re_sents.append(ex_sent)
        new_doc = doc
        return new_doc,np.array(re_sents)






