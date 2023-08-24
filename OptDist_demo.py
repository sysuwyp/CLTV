import os
import numpy as np
import tensorflow as tf
import time
import argparse
import random
from sklearn.metrics import roc_auc_score, mean_absolute_error
import multiprocessing
import queue
import threading

from scipy.stats import rankdata, pearsonr, spearmanr, kendalltau

from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def _normalized_rmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred)) / y_true.mean()

def cumulative_true(
    y_true,
    y_pred
) -> np.ndarray:
  """Calculates cumulative sum of lifetime values over predicted rank.
  Arguments:
    y_true: true lifetime values.
    y_pred: predicted lifetime values.
  Returns:
    res: cumulative sum of lifetime values over predicted rank.
  """
  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
  }).sort_values(
      by='y_pred', ascending=False)

  return (df['y_true'].cumsum() / df['y_true'].sum()).values

def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates gini coefficient over gain charts.
  Arguments:
    df: Each column contains one gain chart. First column must be ground truth.
  Returns:
    gini_result: This dataframe has two columns containing raw and normalized
                 gini coefficient.
  """
  raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
  normalized = raw / raw[0]
  return pd.DataFrame({
      'raw': raw,
      'normalized': normalized
  })[['raw', 'normalized']]
def corr_zero_inflate(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    # labels = rankdata(np.array(labels))
    # predicts = rankdata(np.array(predicts))
    mask1 = labels > 0
    corr0 = spearmanr(labels, predicts)[0]
    corr1 = spearmanr(labels[mask1], predicts[mask1])[0]
    return corr0, corr1


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'





def parse_args():
    parser = argparse.ArgumentParser(description="Run AITM.")
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Batch size.')
    parser.add_argument('--embedding_dim', type=int, default=5,
                        help='Number of embedding dim.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.9,0.7,0.7]',
                        help='Keep probability. 1: no dropout.')
    parser.add_argument('--lamda', type=float, default=1e-6,
                        help='Regularizer weight.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Specify an optimizer type (adam, adagrad, gd, moment).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the results (0, 1 ... any positive integer)')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0, 1 ... any positive integer)')
    parser.add_argument('--prefix', type=str, required=True,
                        help='prefix for model_name path.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which gpu to use.')

    parser.add_argument('--seed', type=int, default=2022,
                        help='args weight.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='args beta.')
    return parser.parse_args()


args = parse_args()
all_columns = ['product_age_group', 'device_type', 'audience_id',
               'product_gender', 'product_brand', 'product_category1', 'product_category2',
               'product_category3', 'product_category4', 'product_category5', 'product_category6',
               'product_category7', 'product_country', 'product_id', 'product_title',
               'partner_id', 'user_id', 'weekday', 'hour']
vocabulary_size = {
    'product_age_group': 12,
    'device_type': 9,
    'audience_id': 18229,
    'product_gender': 18,
    'product_brand': 55984,
    'product_category1': 22,
    'product_category2': 174,
    'product_category3': 1038,
    'product_category4': 1553,
    'product_category5': 842,
    'product_category6': 214,
    'product_category7': 6,
    'product_country': 24,
    'product_id': 1628065,
    'product_title': 790523,
    'partner_id': 312,
    'user_id': 13708489,
    'weekday': 7,
    'hour': 24

}
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def drop_out(_input, dropout, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        output = tf.nn.dropout(_input, keep_prob=dropout)
    else:
        output = _input
    return output


def dnn_layer(dnn_input, mode, dropout, l2_reg=0, batch_norm=True, units=[]):
    dnn_out = dnn_input
    for i in range(len(units)):
        dnn_out = tf.contrib.layers.fully_connected(
            dnn_out, units[i], activation_fn=tf.nn.relu,
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg)
        )
        if mode == tf.estimator.ModeKeys.TRAIN:
            dnn_out = tf.contrib.layers.batch_norm(dnn_out, is_training=True) if batch_norm else dnn_out
            dnn_out = tf.nn.dropout(dnn_out, keep_prob=dropout[i])
    return dnn_out


def print_info(prefix, result, time):
    print(prefix + '[%.1fs]: \n'
                   'convert:     AUC:%.6f\n'
                   'amount:        mae:%.6f.\n'
                   'rmse:        rmse:%.6f.\n'
                   'spr_all:        sprall:%.6f.\n'
                   'spr_positive:        sprpositive:%.6f.\n'
                   'nrmse:        nrmse:%.6f.'
          % tuple([time] + result))



class GeneratorEnqueuer(object):
    """From keras source code training.py
    Builds a queue out of a data generator.
    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading
    """

    def __init__(self, generator, pickle_safe=False):
        self._generator = generator
        self._pickle_safe = pickle_safe
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.finish = False

    def start(self, workers=1, max_q_size=10, wait_time=0.05):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._pickle_safe or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except StopIteration:
                    self.finish = True
                    break
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._pickle_safe:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._pickle_safe:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed()
                    thread = multiprocessing.Process(
                        target=data_generator_task)
                    thread.daemon = True
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except BaseException:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._pickle_safe:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._pickle_safe:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None


def sample_gumbel_01(shape, eps=1e-10):
    """Sample from Gumbel(0, 1) distribution"""
    U = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    # logits: [batch_size, n_classes], unnormalized log-probs
    y = logits + sample_gumbel_01(tf.shape(logits))
    return tf.nn.softmax(y / temperature, axis=-1), y / temperature  # sum of each line equals 1


def sample_gumbel_01(shape, eps=1e-10):
    """Sample from Gumbel(0, 1) distribution"""
    U = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    # logits: [batch_size, n_classes], unnormalized log-probs
    y = logits + sample_gumbel_01(tf.shape(logits))
    return tf.nn.softmax(y / temperature, axis=-1), y / temperature  # sum of each line equals 1


class OptDist(object):
    def __init__(self, vocabulary_size, embedding_dim, epoch, batch_size, learning_rate, lamda,
                 keep_prob, optimizer_type, verbose, early_stop,
                 prefix, random_seed=2020,beta = 1.0):
        # bind params to class
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.lamda = lamda
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        print('dropout:{}'.format(self.keep_prob))
        self.no_dropout = np.array([1 for _ in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.early_stop = early_stop
        self.prefix = prefix
        self.beta = beta
        # init all variables in a tensorflow graph
        self._init_graph_AITM()

    def _init_graph_AITM(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        print('Init raw AITM graph')
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            l2_reg = self.lamda  # tf.contrib.layers.l2_regularizer(self.lamda)

            # Variables init.
            self.cvr_label = tf.placeholder(
                tf.float32, shape=[None, 1], name='convert')
            self.labels_fee = tf.placeholder(
                tf.float32, shape=[None, 1], name='labels_fee')
            self.nb_click = tf.placeholder(
                tf.float32, shape=[None, 1], name="nb_click")
            labels_fee = tf.reshape(self.labels_fee, shape=[-1])
            positive = tf.cast(labels_fee > 0, tf.float32)

            save_fee = positive * labels_fee + (
                    1 - positive) * tf.keras.backend.ones_like(labels_fee)

            self.inputs_placeholder = []
            for column in all_columns:
                self.inputs_placeholder.append(tf.placeholder(
                    tf.int64, shape=[None, 1], name=column))
            self.nb_click = tf.placeholder(
                tf.float32, shape=[None, 1], name="nb_click")

            feature_embedding = []
            num_distribution = 4
            with tf.variable_scope('fee_part'):
                self.weights = self._initialize_weights()
                for column, feature in zip(all_columns, self.inputs_placeholder):
                    embedded = tf.nn.embedding_lookup(self.weights['feature_embeddings_{}'.format(
                        column)], feature)  # [None , 1, K]*num_features
                    feature_embedding.append(embedded)
                feature_embedding = tf.keras.layers.concatenate(feature_embedding)
                feature_embedding = tf.squeeze(feature_embedding, axis=1)
                feature_embedding = tf.concat([feature_embedding, self.nb_click], axis=1)

                # with tf.variable_scope('mu_part'):
                H = tf.layers.dense(feature_embedding, 1024, activation=tf.nn.relu, name="share_layer")
                for i in range(num_distribution):
                    mu_out = tf.keras.layers.Dropout(
                        1 - self.keep_prob[0])(H)
                    mu_cvr_vec = tf.layers.dense(mu_out, 512 // 2, activation=tf.nn.relu,
                                                 name="vector_layer_first" + str(i))
                    mu_cvr_vec = tf.keras.layers.Dropout(
                        1 - self.keep_prob[1])(mu_cvr_vec)
                    mu_cvr_vec = tf.layers.dense(mu_cvr_vec, 256/2, activation=tf.nn.relu,
                                                 name="vector_layer_second" + str(i))
                    mu_cvr_vec = tf.keras.layers.Dropout(
                        1 - self.keep_prob[2])(mu_cvr_vec)

                    mu_cvr_vec = tf.layers.dense(mu_cvr_vec, 128/2, activation=tf.nn.relu,
                                                 name="vector_layer_third" + str(i))
                    cur = tf.layers.dense(mu_cvr_vec, 1, activation=None,

                                          name="output_layer_" + str(i))

                    cur_sigma = tf.layers.dense(mu_cvr_vec, 1, activation=tf.nn.relu,
                                                name="sigma_output_layer_" + str(i))
                    cvr_logits = tf.layers.dense(mu_cvr_vec, 1, activation=tf.nn.sigmoid,
                                                 name="cvr_output_layer_" + str(i))
                    cur_sigma = tf.math.maximum(tf.keras.backend.softplus(cur_sigma), tf.math.sqrt(tf.keras.backend.epsilon()))

                    if i == 0:
                        mu_logitis = cur
                        sigma = cur_sigma
                        ori_pred = cvr_logits

                    else:
                        mu_logitis = tf.concat([mu_logitis, cur], axis=1)
                        sigma = tf.concat([sigma, cur_sigma], axis=1)
                        ori_pred = tf.concat([ori_pred, cvr_logits],axis = 1)

                # mu_logitis = tf.layers.dense(mu_cvr_vec, 5, activation = None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
                # fee_u_logits = tf.reshape(mu_logitis, [-1])

                # mu_logitis = tf.layers.dense(mu_cvr_vec, 5, activation = None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg))
                # fee_u_logits = tf.reshape(mu_logitis, [-1])

            with tf.variable_scope('selector'):
                selector = tf.layers.dense(H, 1024,
                                           activation=tf.nn.relu, name="vector_layer")
                selector = tf.layers.dense(selector, 512,
                                           activation=tf.nn.relu, name="vector_layer2")
                selector = tf.layers.dense(selector, 256,
                                           activation=tf.nn.relu, name="vector_layer3")
                selector = tf.layers.dense(selector, num_distribution, activation=None)
                selector = tf.reshape(selector, [-1, num_distribution])
                # fee_u_logits = tf.where(tf.stop_gradient(pcvr) > 0.3333, fee_u_logits,fee_u_logits2 )#dnn_layer(deep_inputs, mode, dropout, l2_reg, batch_norm, layers)
                # fee_sig_logits = tf.where(tf.stop_gradient(pcvr) > 0.3333, fee_sig_logits, fee_sig_logits2)

            # pred = tf.sigmoid(advance_logits)  # * pred_all_convert
            # pred = tf.sigmoid(advance_logits)  # * pred_all_convert

            res_dict = {}
            fee_sig_logits = sigma  # tf.reshape(fee_sig_logits,shape = [-1])


            for i in range(num_distribution):
                cur_fee_u_logits = tf.reshape(mu_logitis[:, i], shape=[-1])
                cur_fee_sig_logits = tf.reshape(fee_sig_logits[:, i], shape=[-1])

                res_dict['res_' + str(i)] = (
                    tf.exp(tf.minimum(cur_fee_u_logits + cur_fee_sig_logits * cur_fee_sig_logits / 2, 10)))
                if i == 0:
                    final_fee_res = tf.reshape(res_dict['res_' + str(i)], shape=[-1, 1])


                else:
                    final_fee_res = tf.concat([final_fee_res, tf.reshape(res_dict['res_' + str(i)], shape=[-1, 1])],
                                              axis=1)
            gumbel_select,_ = gumbel_softmax_sample(tf.log(tf.nn.softmax(selector,axis = 1)),0.1)
            #l2_loss = tf.reduce_sum((tf.square(tf.reduce_mean(gumbel_select,axis = 0 ) - 1/num_distribution)))

            #l2_loss = tf.nn.l2_loss((selector))#tf.reduce_mean(tf.reduce_sum(selector * selector/2,axis = 1))#


            print("debug print shape:", final_fee_res.shape)
            selector_id = tf.cast(tf.argmax(selector, axis=1), tf.int32)
            self.selector_id = selector_id
            # weighted_evt = pred * tf.reduce_sum(tf.nn.softmax(selector) * final_fee_res, axis=1)
            final_fee_res = tf.gather_nd(final_fee_res, tf.stack((tf.range(tf.shape(final_fee_res)[0],
                                                                           dtype=selector_id.dtype),
                                                                  selector_id),
                                                                 axis=1))

            final_mu = tf.gather_nd(mu_logitis, tf.stack((tf.range(tf.shape(mu_logitis)[0],
                                                                           dtype=selector_id.dtype),
                                                                  selector_id),
                                                                 axis=1))
            final_sigma = tf.gather_nd(fee_sig_logits, tf.stack((tf.range(tf.shape(fee_sig_logits)[0],
                                                                           dtype=selector_id.dtype),
                                                                  selector_id),
                                                                 axis=1))

            pred = tf.gather_nd(ori_pred, tf.stack((tf.range(tf.shape(ori_pred)[0],
                                                             dtype=selector_id.dtype),
                                                    selector_id),
                                                   axis=1))
            # tf.gather(final_fee_res, selector_id, axis = 1)
            # final_fee_res = tf.reshape(final_fee_res, shape = [-1])
            petv = pred * tf.reshape(final_fee_res, shape=[-1])  # + pred/100000
            # tf.gather(final_fee_res, selector_id, axis = 1)
            # final_fee_res = tf.reshape(final_fee_res, shape = [-1])

            self.etv = petv
            self.pred = petv
            self.mu = final_mu
            self.sigma = final_sigma
            self.p = pred

            # Compute the loss.
            # L2
            reg_variables = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.lamda > 0:
                reg_loss = tf.add_n(reg_variables)
            else:
                reg_loss = 0

            # positive = tf.cast(self.labels_fee > 0, tf.float32)

            # loss = tf.reduce_mean(tf.losses.log_loss(labels=tf.reshape(self.cvr_label, shape = [-1]), predictions=pred))
            # save_label = fee_label
            # positive = tf.cast(fee_label > 0, tf.float32)
            # pred = tf.maximum(pred, 0.00000001)
            loss = 0
            d_loss = 0


            #gumbel_label = tf.stop_gradient(gumbel_label)

            # loss = loss +  fee_loss

            # all_fee_loss = tf.concat([fee_loss, fee_loss2, fee_loss3] ,axis = 1)

            save_label = save_fee
            softmax_selector = tf.nn.softmax(selector, axis=-1)
            for i in range(num_distribution):
                cur_fee_u_logits = tf.reshape(mu_logitis[:, i], shape=[-1])
                cur_fee_sig_logits = tf.reshape(fee_sig_logits[:, i], shape=[-1])

                fee_reg = -positive * (
                        -tf.log(tf.sqrt(2 * 3.1415926) * cur_fee_sig_logits * save_label) - (
                        cur_fee_u_logits - tf.log(save_label)) * (
                                cur_fee_u_logits - tf.log(save_label)) / (
                                    2 * cur_fee_sig_logits * cur_fee_sig_logits) + tf.log(
                    tf.maximum(tf.reshape(ori_pred[:, i], shape=[-1]), 0.000001)))

                fee_neg = -(1 - positive) * (tf.log(tf.maximum(1 - tf.reshape(ori_pred[:, i], shape=[-1]), 0.0000000001)))
                fee_loss = fee_reg + fee_neg

                if i == 0:
                    all_fee_loss = tf.reshape(fee_loss, shape=[-1, 1])
                else:
                    all_fee_loss = tf.concat([all_fee_loss, tf.reshape(fee_loss, shape=[-1, 1])], axis=1)
            gumbel_label = tf.nn.softmax(-tf.stop_gradient((all_fee_loss - tf.reshape(tf.reduce_mean(all_fee_loss,axis = 1),shape = [-1,1]))),axis = 1)#gumbel_softmax_sample(-tf.stop_gradient(all_fee_loss), 0.1)
            for i in range(num_distribution):
                d_loss +=   tf.reshape(gumbel_label[:, i], shape=[-1]) *( tf.log(tf.maximum(tf.reshape(gumbel_label[:, i], shape=[-1]),0.0000000001 )) - tf.log(
                    tf.maximum(tf.reshape(softmax_selector[:, i],shape = [-1]),0.0000000001)))

            #final_feeloss_id = tf.cast(tf.argmax(selector, axis=1), tf.int32)
            loss_weight = softmax_selector#gumbel_select#tf.where(gumbel_select > 0.001, gumbel_select, tf.zeros_like(gumbel_select))
            final_feeloss = tf.reduce_sum((loss_weight) * all_fee_loss,axis = 1)
            selector_label = tf.stop_gradient(tf.argmin(all_fee_loss,
                                                        axis=1))  # tf.where(tf.keras.metrics.mean_squared_error(fee_label,pred * advance_cvr_res) > tf.keras.metrics.mean_squared_error(fee_label,pred * advance_cvr_res2), tf.ones_like(cvr_label), tf.zeros_like(cvr_label))
            #min_loss = tf.reduce_min( all_fee_loss,axis = 1)  # tf.reduce_mean(tf.gather(all_fee_loss, final_feeloss_id,axis = 1))
                                       #                 axis=1))  # tf.where(tf.keras.metrics.mean_squared_error(fee_label,pred * advance_cvr_res) > tf.keras.metrics.mean_squared_error(fee_label,pred * advance_cvr_res2), tf.ones_like(cvr_label), tf.zeros_like(cvr_label))
            #min_loss =  tf.gather_nd(all_fee_loss, tf.stack((tf.range(tf.shape(selector_label)[0],
            #                                                         dtype=tf.int32),
            #                                                tf.cast(selector_label, tf.int32)),
            #                                               axis=1))  # tf.reduce_mean(tf.gather(all_fee_loss, final_feeloss_id,axis = 1))
            weighted_selector_loss = 0
            sigmoid_select = softmax_selector
            minlabel = tf.one_hot(selector_label,depth=num_distribution)

            min_loss = 0
            for i in range(num_distribution):
                cur_min = tf.cast(tf.reshape(minlabel[:,i],shape = [-1]),tf.float32)
                cur_dis = tf.cast(tf.reshape(sigmoid_select[:,i],shape = [-1]),tf.float32)
                cur_loss = tf.reduce_mean( -cur_min * ((1 -  cur_dis) ** 2) * tf.log(cur_dis)) #selector_loss #tf.reshape(selector_loss[:,i])
                weighted_selector_loss += cur_loss #tf.reduce_mean(cur_min_loss)#/(weighted_num) #* tf.maximum(5 * tf.reduce_mean(cur_min),1/num_distribution)#tf.reduce_sum(cur_min * cur_loss)/tf.maximum(tf.reduce_sum(cur_min),1.0)/4



            min_loss = tf.reduce_min( all_fee_loss,axis = 1)  # tf.reduce_mean(tf.gather(all_fee_loss, final_feeloss_id,axis = 1))
            min_loss = tf.reduce_mean(min_loss)
            # s_vars = [var for var in t_vars if var.name.startswith('selector')]
            #overall_fee_loss =  self.beta * final_loss
            loss += tf.reduce_mean(final_feeloss) + self.beta * min_loss # + tf.reduce_mean(min_loss) +  tf.reduce_mean(final_feeloss)
            # loss2 = selector_loss

            # -------label_constraint--------

            self.loss = loss  # tf.reduce_mean(fee_loss)
            self.selector_loss = tf.concat( [tf.reduce_mean(tf.one_hot(selector_id,depth= num_distribution),axis = 0), tf.reduce_mean(tf.one_hot(selector_label,depth= num_distribution),axis = 0)],axis = 0) #tf.reduce_sum(tf.reshape(tf.reduce_max(tf.one_hot(selector_label,depth= num_distribution),axis = 0),shape = [-1]))#(selector_loss + tf.reduce_mean( d_loss ))#(selector_loss + tf.reduce_mean( d_loss ))
            self.loss +=  (weighted_selector_loss + tf.reduce_mean( d_loss )) #+ l2_loss #+ tf.reduce_mean(final_feeloss-tf.stop_gradient(min_loss))

            #self.softmax_loss =  (selector_loss + tf.reduce_mean( d_loss ))


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Optimizer.
                if self.optimizer_type == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                                                            beta2=0.999, epsilon=1e-8).minimize(self.loss)

                    #self.optimizer_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9,
                    #                                        beta2=0.999, epsilon=1e-8).minimize(self.softmax_loss)


                elif self.optimizer_type == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                               initial_accumulator_value=1e-8).minimize(self.loss)
                elif self.optimizer_type == 'gd':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                        self.loss)
                elif self.optimizer_type == 'moment':
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver(var_list=tf.global_variables())
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.InteractiveSession(
                config=tf.ConfigProto(
                    gpu_options=gpu_options))
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        '''
        initialize parameters.
        '''
        all_weights = dict()
        l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)
        # attention
        all_weights['attention_w1'] = tf.get_variable(
            initializer=tf.random_normal(
                shape=[32, 32],
                mean=0.0,
                stddev=0.01),
            regularizer=l2_reg, name='attention_w1')  # k * k
        all_weights['attention_w2'] = tf.get_variable(
            initializer=tf.random_normal(
                shape=[32, 32],
                mean=0.0,
                stddev=0.01),
            regularizer=l2_reg, name='attention_w2')  # k * k
        all_weights['attention_w3'] = tf.get_variable(
            initializer=tf.random_normal(
                shape=[32, 32],
                mean=0.0,
                stddev=0.01),
            regularizer=l2_reg, name='attention_w3')  # k * k
        # embedding
        for column in all_columns:
            all_weights['feature_embeddings_{}'.format(column)] = tf.get_variable(
                initializer=tf.random_normal(
                    shape=[
                        vocabulary_size[column],
                        self.embedding_dim],
                    mean=0.0,
                    stddev=0.01),
                regularizer=l2_reg, name='feature_embeddings_{}'.format(column))  # vocabulary_size * K
        return all_weights

    def fit_on_batch(self, data):
        '''
        Fit on a batch data.
        :param data: a batch data.
        :return: The LogLoss.
        '''
        train_ids = {}
        for column_name, column_placeholder in zip(
                all_columns, self.inputs_placeholder):
            train_ids[column_placeholder] = data['ids_{}'.format(column_name)]
        train_ids[self.nb_click] = data['ids_{}'.format("nb_click")]
        feed_dict = {
            self.cvr_label: data['convert'],
            self.labels_fee: data['amount']}
        feed_dict.update(train_ids)

        loss, selector_loss, ad_1 = self.sess.run(
            (self.loss, self.selector_loss, self.optimizer), feed_dict=feed_dict)
        return loss, selector_loss

    def fit(self, train_path, dev_path,
            pickle_safe=False, max_q_size=20, workers=1):
        '''
        Fit the train data.
        :param train_path: train path.
        :param dev_path:  validation path.
        :param pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
        :param max_q_size: maximum size for the generator queue
        :param workers: maximum number of processes to spin up
                when using process based threading
        :return: None
        '''
        max_acc = -np.inf
        best_epoch = 0
        earlystop_count = 0
        enqueuer = None
        wait_time = 0.001  # in seconds
        for epoch in range(self.epoch):
            tf.keras.backend.set_learning_phase(1)
            try:
                train_gen = self.iterator(train_path, shuffle=True)
                enqueuer = GeneratorEnqueuer(
                    train_gen, pickle_safe=pickle_safe)
                enqueuer.start(max_q_size=max_q_size, workers=workers)
                t1 = time.time()
                train_loss = 0.
                selector_loss = 0.0
                nb_sample = 0
                i = 0
                while True:
                    # get a batch
                    generator_output = None
                    while enqueuer.is_running():
                        if not enqueuer.queue.empty():
                            generator_output = enqueuer.queue.get()
                            break
                        elif enqueuer.finish:
                            break
                        else:
                            time.sleep(wait_time)
                    # Fit training, return loss...
                    if generator_output is None:  # epoch end
                        break
                    nb_sample += len(generator_output['convert'])
                    if i % 1 == 0:
                        cur_train_loss, cur_selector_loss = self.fit_on_batch(generator_output)
                    else:
                        cur_train_loss, cur_selector_loss = self.fit_on_batch(generator_output)
                    train_loss += cur_train_loss
                    selector_loss = cur_selector_loss  # * 20
                    #print("selector:", selector_loss)
                    if self.verbose > 0:
                        if (i + 1) % 10000 == 0:
                            print('[%d]Train loss on step %d: %.6f, %.4f' %
                                  (nb_sample, (i + 1), train_loss / (i + 1), 1 / ( 1)))
                    i += 1
                # validation
                tf.keras.backend.set_learning_phase(0)
                t2 = time.time()
                dev_gen = self.iterator(dev_path)
                true_pred = self.evaluate_generator(
                    dev_gen, max_q_size=max_q_size, workers=workers, pickle_safe=pickle_safe)
                valid_result = self.evaluate(true_pred)

                if self.verbose > 0:
                    print_info(
                        "Epoch %d [%.1f s]\t Dev" %
                        (epoch + 1, t2 - t1), valid_result, time.time() - t2)
                if self.early_stop > 0:
                    acc = valid_result[0] - valid_result[1]
                    if max_acc >= acc:  # no gain
                        earlystop_count += 1
                    else:
                        self.save_path = self.saver.save(self.sess,
                                                         save_path='./best_model_{}.model'.format(
                                                             self.prefix),
                                                         latest_filename='check_point_{}'.format(self.prefix))
                        max_acc = acc
                        best_epoch = epoch + 1
                        earlystop_count = 0
                    if earlystop_count >= self.early_stop:
                        if self.verbose > 0:
                            print(
                                "Early stop at Epoch %d based on the best validation Epoch %d." % (
                                    epoch + 1, best_epoch))
                        break

            finally:
                if enqueuer is not None:
                    enqueuer.stop()

    def evaluate_generator(self, generator, max_q_size=20,
                           workers=1, pickle_safe=False):
        '''
        See GeneratorEnqueuer Class about the following params.
        :param generator: the generator which return the data.
        :param max_q_size: maximum size for the generator queue
        :param workers: maximum number of processes to spin up
                when using process based threading
        :param pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
        :return: true labels, prediction probabilities.
        '''
        wait_time = 0.01
        enqueuer = None
        dev_y_true_convert = []
        dev_y_true_amount = []
        dev_y_pred_convert = []
        dev_y_pred_amount = []
        selections = []
        mus = []
        sigmas= []
        ps = []
        try:
            enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
            enqueuer.start(workers=workers, max_q_size=max_q_size)
            nb_dev = 0
            while True:
                dev_batch = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        dev_batch = enqueuer.queue.get()
                        break
                    elif enqueuer.finish:
                        break
                    else:
                        time.sleep(wait_time)
                # Fit training, return loss...
                if dev_batch is None:
                    break
                nb_dev += len(dev_batch['convert'])
                train_ids = {}
                for column_name, column_placeholder in zip(
                        all_columns, self.inputs_placeholder):
                    train_ids[column_placeholder] = dev_batch['ids_{}'.format(
                        column_name)]
                train_ids[self.nb_click] = dev_batch['ids_{}'.format(
                    "nb_click")]
                feed_dict = {
                    self.cvr_label: dev_batch['convert'],
                    self.labels_fee: dev_batch['amount']}
                feed_dict.update(train_ids)
                predictions = self.sess.run(
                    [self.pred, self.etv,self.selector_id,self.mu,self.sigma,self.p], feed_dict=feed_dict)
                dev_y_true_convert += list(dev_batch['convert'])
                dev_y_true_amount += list(dev_batch['amount'])
                dev_y_pred_convert += list(predictions[0])
                dev_y_pred_amount += list(predictions[1])
                selections +=  list(predictions[2])
                mus += list(predictions[3])
                sigmas += list(predictions[4])
                ps += list(predictions[5])

            # to row vectors
            dev_y_true_convert = np.reshape(dev_y_true_convert, (-1,))
            dev_y_true_amount = np.reshape(dev_y_true_amount, (-1,))
            dev_y_pred_convert = np.reshape(dev_y_pred_convert, (-1,))
            dev_y_pred_amount = np.reshape(dev_y_pred_amount, (-1,))
            selection = np.reshape(selections,(-1,))
            mus = np.reshape(mus,(-1,))
            sigmas = np.reshape(sigmas,(-1,))
            ps = np.reshape(ps,(-1,))



            print('Evaluate on %d samples.' % nb_dev)
        finally:
            if enqueuer is not None:
                enqueuer.stop()

        return {'convert_true': dev_y_true_convert, 'convert_pred': dev_y_pred_convert,
                'amount_true': dev_y_true_amount, 'amount_pred': dev_y_pred_amount,"selection": selection, "mus": mus, "sigmas":sigmas, "ps":ps}

    def iterator(self, path, shuffle=False):
        '''
        Generator of data.
        :param path: data path.
        :param shuffle: whether to shuffle the data. It should be True for training set.
        :return: a batch data.
        '''
        prefetch = 50  # prefetch number of batches.
        batch_lines = []
        with open(path, 'r') as fr:
            lines = []
            # remove csv header
            fr.readline()
            for prefetch_line in fr:
                lines.append(prefetch_line)
                if len(lines) >= self.batch_size * prefetch:
                    if shuffle:
                        random.shuffle(lines)
                    for line in lines:
                        batch_lines.append(line.split(','))
                        if len(batch_lines) >= self.batch_size:
                            batch_array = np.array(batch_lines)
                            batch_lines = []
                            batch_data = {}
                            batch_data['convert'] = batch_array[:,
                                                    0:1].astype(np.float64)
                            batch_data['amount'] = batch_array[:,
                                                   1:2].astype(np.float64)
                            for i, column in enumerate(all_columns):
                                batch_data['ids_{}'.format(
                                    column)] = batch_array[:, i + 2:i + 3].astype(np.int64)
                            batch_data['ids_nb_click'.format(
                                column)] = batch_array[:, len(all_columns) + 2:len(all_columns) + 3].astype(np.float32)
                            yield batch_data
                    lines = []
            if 0 < len(lines) < self.batch_size * prefetch:
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    batch_lines.append(line.split(','))
                    if len(batch_lines) >= self.batch_size:
                        batch_array = np.array(batch_lines)
                        batch_lines = []
                        batch_data = {}
                        batch_data['convert'] = batch_array[:,
                                                0:1].astype(np.float64)
                        batch_data['amount'] = batch_array[:,
                                               1:2].astype(np.float64)
                        for i, column in enumerate(
                                all_columns):
                            batch_data['ids_{}'.format(
                                column)] = batch_array[:, i + 2:i + 3].astype(np.int64)
                        batch_data['ids_nb_click'.format(
                            column)] = batch_array[:, len(all_columns) + 2:len(all_columns) + 3].astype(np.float32)
                        yield batch_data
                if 0 < len(batch_lines) < self.batch_size:
                    batch_array = np.array(batch_lines)
                    batch_data = {}
                    batch_data['convert'] = batch_array[:,
                                            0:1].astype(np.float64)
                    batch_data['amount'] = batch_array[:,
                                           1:2].astype(np.float64)
                    for i, column in enumerate(all_columns):
                        batch_data['ids_{}'.format(
                            column)] = batch_array[:, i + 2:i + 3].astype(np.int64)
                    batch_data['ids_nb_click'.format(
                        column)] = batch_array[:, len(all_columns) + 2:len(all_columns) + 3].astype(np.float32)
                    yield batch_data

    def evaluate(self, true_pred):
        '''
        Evaluation Metrics.
        :param true_pred: dict that contains the label and prediction.
        :return: click_auc, purchase_auc
        '''
        auc_convert = roc_auc_score(
            y_true=true_pred['convert_true'],
            y_score=true_pred['convert_pred'])
        mae_amount = mean_absolute_error(
            true_pred['amount_true'],
            true_pred['amount_pred'])
        rmse = sqrt(mean_squared_error(true_pred['amount_true'],
            true_pred['amount_pred']))
        spr_all,spr_positive = corr_zero_inflate(true_pred['amount_true'],true_pred['amount_pred'])

        nrmse = _normalized_rmse(y_pred=true_pred['amount_pred'], y_true= true_pred['amount_true'] )

        gain = pd.DataFrame({
            'lorenz': cumulative_true(true_pred['amount_true'],true_pred['amount_true']),
            'model': cumulative_true(true_pred['amount_true'], true_pred['amount_pred'])
        })
        num_customers = np.float32(gain.shape[0])
        gain['cumulative_customer'] = (np.arange(num_customers) + 1.) / num_customers
        gini = gini_from_gain(gain[['lorenz', 'model']])
        print(gini)
        selection = list(true_pred['selection'])
        for i in range(4):
            count = selection.count(i)


            print(i, 'selection num:', count)
        print("save_to_csv")
        std_df = pd.DataFrame({
            'label':true_pred['amount_true'],
            'prediction':true_pred['amount_pred'],
            'selection': true_pred['selection'],
            'mus': true_pred['mus'],
            'simgas':true_pred['sigmas'],
            'ps':true_pred['ps']
        })
        std_df.to_csv("./save_cluster_criteo_obs.csv",encoding = "utf-8")

        return [auc_convert, mae_amount,rmse,spr_all,spr_positive,nrmse]


    def evaluate(self, true_pred):
        '''
        Evaluation Metrics.
        :param true_pred: dict that contains the label and prediction.
        :return: click_auc, purchase_auc
        '''
        auc_convert = roc_auc_score(
            y_true=true_pred['convert_true'],
            y_score=true_pred['convert_pred'])
        mae_amount = mean_absolute_error(
            true_pred['amount_true'],
            true_pred['amount_pred'])
        rmse = sqrt(mean_squared_error(true_pred['amount_true'],
            true_pred['amount_pred']))
        spr_all,spr_positive = corr_zero_inflate(true_pred['amount_true'],true_pred['amount_pred'])

        nrmse = _normalized_rmse(y_pred=true_pred['amount_pred'], y_true= true_pred['amount_true'] )

        gain = pd.DataFrame({
            'lorenz': cumulative_true(true_pred['amount_true'],true_pred['amount_true']),
            'model': cumulative_true(true_pred['amount_true'], true_pred['amount_pred'])
        })
        num_customers = np.float32(gain.shape[0])
        gain['cumulative_customer'] = (np.arange(num_customers) + 1.) / num_customers
        gini = gini_from_gain(gain[['lorenz', 'model']])
        print(gini)

        labels = true_pred['amount_true']
        probs = true_pred['amount_pred']
        not_none_labels = [l for l, p in zip(labels, probs) if l > 0.0]
        not_none_probs = [p for l, p in zip(labels, probs) if l > 0.0]
        not_non_mse = sqrt(mean_squared_error(np.array(not_none_labels), np.array(not_none_probs)))
        not_non_mae = mean_absolute_error(not_none_labels, not_none_probs)


        gain_pos = pd.DataFrame({
            'lorenz': cumulative_true(not_none_labels,not_none_labels),
            'model': cumulative_true(not_none_labels,not_none_probs)
        })
        num_customers_pos = np.float32(gain_pos.shape[0])
        gain_pos['cumulative_customer'] = (np.arange(num_customers_pos) + 1.) / num_customers_pos
        gini_pos = gini_from_gain(gain_pos[['lorenz', 'model']])
        print("gini positive:")
        print(gini_pos)
        print("save_to_csv")

        std_df = pd.DataFrame({
            'label':true_pred['amount_true'],
            'prediction':true_pred['amount_pred'],
            'selection': true_pred['selection'],
            'mus': true_pred['mus'],
            'simgas':true_pred['sigmas'],
            'ps':true_pred['ps']
        })
        std_df.to_csv("./save_cluster_criteo_obs.csv",encoding = "utf-8")

        return [auc_convert, mae_amount,rmse,spr_all,spr_positive,nrmse]


if __name__ == '__main__':
    data_path = './'
    train_path, dev_path, test_path = os.path.join(data_path, 'train_df.csv'), \
                                      os.path.join(
                                          data_path, 'val_df.csv'), os.path.join(
        data_path, 'test_df.csv')
    max_q_size = 50
    workers = 1
    pickle_safe = False

    args.prefix = args.prefix.replace('"', '')
    print(eval(args.keep_prob))
    # Training
    t1 = time.time()
    model = OptDist(vocabulary_size=vocabulary_size, embedding_dim=args.embedding_dim,
                 epoch=args.epoch,
                 batch_size=args.batch_size, learning_rate=args.lr, lamda=args.lamda,
                 keep_prob=eval(args.keep_prob), optimizer_type=args.optimizer, verbose=args.verbose,
                 early_stop=args.early_stop,
                 prefix=args.prefix, random_seed=args.seed,beta=args.beta)
    model.fit(train_path, dev_path, pickle_safe=pickle_safe, max_q_size=max_q_size,
              workers=workers)
    # restore the best model
    model.saver.restore(model.sess, save_path=model.save_path)
    tf.keras.backend.set_learning_phase(0)

    # Test
    t = time.time()
    test_gen = model.iterator(test_path)
    true_pred = model.evaluate_generator(test_gen, max_q_size=max_q_size,
                                         workers=workers,
                                         pickle_safe=pickle_safe)
    test_result = model.evaluate(true_pred)
    print_info('Test', test_result, time.time() - t)





