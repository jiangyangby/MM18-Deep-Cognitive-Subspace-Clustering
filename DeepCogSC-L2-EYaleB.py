import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import scipy.io as sio
from utils import thrC, post_proC, err_rate


class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0,
                 re_constant2=1.0, batch_size=200, reg=None, denoise=False,
                 model_path=None, restore_path=None, logs_path='./logs/'):
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

        if not denoise:
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
        reconst_cost = tf.pow(tf.subtract(self.x_r, self.x), 2.0)
        self.reconst_cost = 0.5 * tf.reduce_sum(reconst_cost)
        self.reconst_cost_list = 0.5 * \
            tf.reduce_sum(reconst_cost, axis=[1, 2, 3])
        tf.summary.scalar("recons_loss", self.reconst_cost)

        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef, 2.0))
        # self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))  # l1 reg
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        selfexpress_cost = tf.pow(tf.subtract(z_c, z), 2.0)
        self.selfexpress_losses = 0.5 * tf.reduce_sum(selfexpress_cost)
        self.selfexpress_cost_list = 0.5 * \
            tf.reduce_sum(selfexpress_cost, axis=[1])
        tf.summary.scalar("selfexpress_loss", re_constant2 *
                          self.selfexpress_losses)

        self.loss = self.reconst_cost + reg_constant1 * \
            self.reg_losses + re_constant2 * self.selfexpress_losses
        self.sp_loss_list = self.reconst_cost_list + \
            re_constant2 * self.selfexpress_cost_list
        self.sp_loss = tf.reduce_sum(tf.multiply(
            self.v, self.sp_loss_list) + reg_constant1 * self.reg_losses)

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
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(
            tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0], self.n_hidden[1]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b1'] = tf.Variable(
            tf.zeros([self.n_hidden[1]], dtype=tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1], self.n_hidden[2]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b2'] = tf.Variable(
            tf.zeros([self.n_hidden[2]], dtype=tf.float32))

        all_weights['Coef'] = tf.Variable(
            1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')

        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1], self.n_hidden[2]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['dec_b0'] = tf.Variable(
            tf.zeros([self.n_hidden[1]], dtype=tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0], self.n_hidden[1]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['dec_b1'] = tf.Variable(
            tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32))

        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[
                                1, 2, 2, 1], padding='SAME'), weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        shapes.append(layer1.get_shape().as_list())
        layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[
                                1, 2, 2, 1], padding='SAME'), weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        shapes.append(layer2.get_shape().as_list())
        layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[
                                1, 2, 2, 1], padding='SAME'), weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)
        return layer3, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(self.x)[0], shape_de1[1], shape_de1[2], shape_de1[3]]),
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack([tf.shape(self.x)[0], shape_de2[1], shape_de2[2], shape_de2[3]]),
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)
        shape_de3 = shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack([tf.shape(self.x)[0], shape_de3[1], shape_de3[2], shape_de3[3]]),
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b2'])
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


def train_face(X, y, CAE, num_class, batch_size):
    lr = 0.9e-4
    max_step = 50 + num_class * 25  # 100+num_class*20
    alpha = max(0.4 - (num_class - 1) / 10 * 0.1, 0.1)
    print('alpha=%f' % alpha)

    error_list = []
    for i in range(39 - num_class):
        X_batch = X[64 * i:64 * (i + num_class), :].astype(float)
        y_batch = y[64 * i:64 * (i + num_class)]
        y_batch = y_batch - y_batch.min() + 1
        y_batch = np.squeeze(y_batch)

        # err = train(i, X_batch, y_batch, CAE, lr, alpha, max_step)
        err = sp_train(i, X_batch, y_batch, CAE, lr, alpha, 64, 0, 6, 13)
        error_list.append(err)

    mean_err = np.mean(error_list)
    median_err = np.median(error_list)
    print("%d subjects:" % num_class)
    print("Mean: %.4f%%" % (mean_err * 100))
    print("Median: %.4f%%" % (median_err * 100))
    print(error_list)

    return mean_err, median_err


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
        # print('cost list:', cost_list)
        lambda_t = max(sorted(cost_list)[n - 1], (1 + tou) * lambda_t)
        print('lambda_t=%f' % lambda_t)
        for i in range(outers):
            V = [math.exp(-cost / lambda_t) for cost in cost_list]
            for j in range(inners):
                cost, cost_list, Coef = CAE.sp_partial_fit(X, V, lr)
            print("outer: %d" % i, "inner: %d" % j, "cost: %.8f" %
                  (cost / float(n)))
    Coef = thrC(Coef, alpha)
    y_pred, _ = post_proC(Coef, y.max(), 10, 3.5)
    err, y_map = err_rate(y, y_pred)
    print("experiment: %d" % iteration, "cost: %.8f" %
          (cost / float(num_samples)), "error rate: %.4f%%" % (err * 100))
    return err


def load_data(file):
    data = sio.loadmat(file)
    img = data['Y']  # (2016, 64, 38)
    # (2016, 64, 38) -> (42, 48, 2432) -> (2432, 48, 42)
    X = np.array([np.reshape(img[:, j, i], [42, 48])
                  for i in range(img.shape[2])
                  for j in range(img.shape[1])]).transpose([0, 2, 1])
    y = np.array([i for i in range(img.shape[2])
                  for j in range(img.shape[1])])
    X = np.expand_dims(X, 3)
    return X, y


if __name__ == '__main__':
    # load face images and labels
    X, y = load_data('./data/YaleBCrop025.mat')

    # face image clustering
    n_input = [48, 42]
    kernel_size = [5, 3, 3]
    n_hidden = [10, 20, 30]

    all_subjects = [38]

    avg = []
    med = []

    model_path = './models/model-YaleB.ckpt'
    restore_path = './models/model-YaleB.ckpt'
    logs_path = './logs/'
    reg1 = 1.0

    for num_class in all_subjects:
        batch_size = num_class * 64
        reg2 = 1.0 * 10 ** (num_class / 10.0 - 3.0)
        tf.reset_default_graph()
        CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1,
                     re_constant2=reg2, kernel_size=kernel_size,
                     batch_size=batch_size, model_path=model_path,
                     restore_path=restore_path, logs_path=logs_path)

        avg_i, med_i = train_face(
            X, y, CAE, num_class, batch_size)
        avg.append(avg_i)
        med.append(med_i)

    for num_class, mean, median in zip(all_subjects, avg, med):
        print('%d subjects:' % num_class)
        print('Mean: %.4f%%' % (mean * 100),
              'Median: %.4f%%' % (median * 100))
