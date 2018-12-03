# Code Authors: Pan Ji,     University of Adelaide,         pan.ji@adelaide.edu.au
#               Tong Zhang, Australian National University, tong.zhang@anu.edu.au
# Copyright Reserved!
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
import scipy.io as sio
from utils import thrC, post_proC, err_rate


class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0,
                 re_constant2=1.0, batch_size=200, reg=None,
                 denoise=False, model_path=None, restore_path=None,
                 logs_path='./logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0

        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], 1])
        self.v = tf.placeholder(tf.float32, [None])
        self.learning_rate = tf.placeholder(tf.float32, [])

        weights = self._initialize_weights()

        if denoise is False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)
        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)

        z = tf.reshape(latent, [batch_size, -1])
        Coef = weights['Coef']
        z_c = tf.matmul(Coef, z)
        self.Coef = Coef
        latent_c = tf.reshape(z_c, tf.shape(latent))
        self.z = z
        self.x_r = self.decoder(latent_c, weights, shape)

        # l_2 reconstruction loss
        reconst_cost = tf.pow(tf.subtract(self.x_r, self.x), 2.0)  # l2
        self.reconst_cost = 0.5 * tf.reduce_sum(reconst_cost)
        self.reconst_cost_list = 0.5 * \
            tf.reduce_sum(reconst_cost, axis=[1, 2, 3])
        tf.summary.scalar("recons_loss", self.reconst_cost)

        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef, 2.0))  # l2 reg
        # self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))  # l1 reg
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        selfexpress_cost = tf.pow(tf.subtract(z_c, z), 2.0)
        self.selfexpress_losses = 0.5 * tf.reduce_sum(selfexpress_cost)
        self.selfexpress_cost_list = 0.5 * \
            tf.reduce_sum(selfexpress_cost, axis=[1])
        tf.summary.scalar("selfexpress_loss", re_constant2 *
                          self.selfexpress_losses)

        Coef_diag = tf.diag_part(self.Coef)
        diag_cost_list = tf.pow(Coef_diag, 2.0)
        diag_loss = tf.reduce_sum(diag_cost_list)

        self.loss = self.reconst_cost + reg_constant1 * \
            self.reg_losses + re_constant2 * self.selfexpress_losses
        self.sp_loss_list = self.reconst_cost_list + \
            re_constant2 * self.selfexpress_cost_list + diag_cost_list
        self.sp_loss = tf.reduce_sum(tf.multiply(
            self.v, self.sp_loss_list) + reg_constant1 * self.reg_losses + diag_loss)

        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)
        self.sp_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.sp_loss)

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver(
            [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        self.summary_writer = tf.summary.FileWriter(
            logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef'] = tf.Variable(
            0 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')

        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(
            tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        for iter_i in range(1, n_layers):
            enc_name_wi = 'enc_w' + str(iter_i)
            all_weights[enc_name_wi] = tf.get_variable(enc_name_wi, shape=[self.kernel_size[iter_i], self.kernel_size[iter_i], self.n_hidden[iter_i - 1],
                                                                           self.n_hidden[iter_i]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            enc_name_bi = 'enc_b' + str(iter_i)
            all_weights[enc_name_bi] = tf.Variable(
                tf.zeros([self.n_hidden[iter_i]], dtype=tf.float32))

        for iter_i in range(1, n_layers):
            dec_name_wi = 'dec_w' + str(iter_i - 1)
            all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[n_layers - iter_i], self.kernel_size[n_layers - iter_i],
                                                                           self.n_hidden[n_layers - iter_i - 1], self.n_hidden[n_layers - iter_i]],
                                                       initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
            dec_name_bi = 'dec_b' + str(iter_i - 1)
            all_weights[dec_name_bi] = tf.Variable(tf.zeros(
                [self.n_hidden[n_layers - iter_i - 1]], dtype=tf.float32))

        dec_name_wi = 'dec_w' + str(n_layers - 1)
        all_weights[dec_name_wi] = tf.get_variable(dec_name_wi, shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                   initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        dec_name_bi = 'dec_b' + str(n_layers - 1)
        all_weights[dec_name_bi] = tf.Variable(
            tf.zeros([1], dtype=tf.float32))

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        shapes.append(x.get_shape().as_list())
        layeri = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[
                                1, 2, 2, 1], padding='SAME'), weights['enc_b0'])
        layeri = tf.nn.relu(layeri)
        shapes.append(layeri.get_shape().as_list())

        for iter_i in range(1, len(self.n_hidden)):
            layeri = tf.nn.bias_add(tf.nn.conv2d(layeri, weights['enc_w' + str(iter_i)], strides=[
                                    1, 2, 2, 1], padding='SAME'), weights['enc_b' + str(iter_i)])
            layeri = tf.nn.relu(layeri)
            shapes.append(layeri.get_shape().as_list())

        layer3 = layeri
        return layer3, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        n_layers = len(self.n_hidden)
        layer3 = z
        for iter_i in range(n_layers):
            shape_de = shapes[n_layers - iter_i - 1]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer3, weights['dec_w' + str(iter_i)], tf.stack([tf.shape(self.x)[0], shape_de[1], shape_de[2], shape_de[3]]),
                                                   strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b' + str(iter_i)])
            layer3 = tf.nn.relu(layer3)
        return layer3

    def partial_fit(self, X, lr):
        cost, summary, _, Coef = self.sess.run(
            (self.reconst_cost, self.merged_summary_op, self.optimizer, self.Coef),
            feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef

    def sp_partial_fit(self, X, V, lr):
        cost, cost_list, summary, _, Coef = self.sess.run(
            (self.sp_loss, self.sp_loss_list,
             self.merged_summary_op, self.sp_optimizer, self.Coef),
            feed_dict={self.x: X, self.v: V, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, cost_list, Coef

    def inference(self, X):
        cost_list = self.sess.run(
            self.sp_loss_list, feed_dict={self.x: X})
        return cost_list

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")


def sp_train(iteration, X, y, CAE, lr, alpha, batch_size,
             init_n, outers, inners):
    CAE.initlization()
    CAE.restore()  # restore from pre-trained model
    num_samples = X.shape[0]
    print('num samples=%d' % num_samples)
    Ns = list(range(int(num_samples / 2), num_samples + 1, batch_size))
    print('Ns: ', Ns)
    cost_list = CAE.inference(X)
    lambda_t = 0
    tou = 0.15
    for n in Ns:
        print('N=%d' % n)
        lambda_t = max(sorted(cost_list)[n - 1], (1 + tou) * lambda_t)
        print('lambda_t=%f' % lambda_t)
        for i in range(outers):
            V = [math.exp(-cost / lambda_t) for cost in cost_list]
            for j in range(inners):
                cost, cost_list, Coef = CAE.sp_partial_fit(X, V, lr)
            print("outer: %d" % i, "inner: %d" % j, "cost: %.8f" %
                  (cost / float(n)))
    Coef = thrC(Coef, alpha)
    y_pred, _ = post_proC(Coef, y.max(), 3, 1)
    err, y_new = err_rate(y, y_pred)
    print("experiment: %d" % iteration, "cost: %.8f" %
          (cost / float(num_samples)), "error rate: %.4f%%" % (err * 100))
    return err, Coef


def train_face(X, y, CAE, num_class):
    lr = 5.5e-4
    alpha = 0.142
    print('alpha=%f' % alpha)
    outers = 6
    inners = 6

    X_batch = np.array(X).astype(float)
    y_batch = np.array(y)
    y_batch = y_batch - y_batch.min() + 1
    y_batch = np.squeeze(y_batch)

    err, Coef = sp_train(0, X_batch,
                         y_batch, CAE, lr, alpha,
                         10, 0, outers, inners)
    print("%d subjects:" % num_class)
    print("Err: %.4f%%" % (err * 100))

    return err


if __name__ == '__main__':

    # load face images and labels
    data = sio.loadmat('./data/ORL_32x32.mat')
    X = data['fea']
    y = data['gnd']

    # face image clustering
    n_input = [32, 32]
    kernel_size = [3, 3, 3]
    n_hidden = [3, 3, 5]

    X = np.reshape(X, [X.shape[0], n_input[0], n_input[1], 1])

    model_path = './models/model-ORL.ckpt'
    restore_path = './models/model-ORL.ckpt'
    logs_path = './logs/'
    reg1 = 1.0
    reg2 = 0.2
    num_class = 40

    batch_size = num_class * 10
    tf.reset_default_graph()
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1,
                 re_constant2=reg2, kernel_size=kernel_size,
                 batch_size=batch_size, model_path=model_path,
                 restore_path=restore_path, logs_path=logs_path)

    err = train_face(X, y, CAE, num_class)

    print('%d subjects:' % num_class)
    print('Err: %.4f%%' % (err * 100))
