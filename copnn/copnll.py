from os import name
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
from scipy import special
import tensorflow.keras.backend as K


class COPNLL(Layer):
    """COPNN Negative Log Likelihood Loss Layer"""

    def __init__(self, mode, sig2e, sig2bs, rhos = [], est_cors = [], Z_non_linear=False, dist_matrix=None, lengthscale=None, distribution=None):
        super(COPNLL, self).__init__(dynamic=False)
        self.sig2bs = tf.Variable(
            sig2bs, name='sig2bs', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
        self.Z_non_linear = Z_non_linear
        self.mode = mode
        self.rhos = None
        self.est_cors = None
        self.dist_matrix = None
        self.lengthscale = None
        if self.mode in ['categorical', 'longitudinal', 'spatial', 'spatial_embedded', 'spatial_and_categoricals', 'mme']:
            self.sig2e = tf.Variable(
                sig2e, name='sig2e', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
            if self.mode in ['spatial', 'spatial_and_categoricals', 'mme']:
                self.dist_matrix = dist_matrix
                self.max_loc = dist_matrix.shape[1] - 1
                self.spatial_delta = int(0.0 * dist_matrix.shape[1])
                self.lengthscale = tf.Variable(
                    lengthscale, name='lengthscale', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
        if self.mode == 'longitudinal':
            if len(est_cors) > 0:
                self.rhos = tf.Variable(
                    rhos, name='rhos', constraint=lambda x: tf.clip_by_value(x, -1.0, 1.0))
            self.est_cors = est_cors
        if self.mode == 'glmm':
            self.nGQ = 5
            self.x_ks, self.w_ks = np.polynomial.hermite.hermgauss(self.nGQ)
        self.distribution = distribution

    def get_vars(self):
        if self.mode in ['categorical', 'spatial_embedded', 'spatial_and_categoricals', 'mme']:
            return self.sig2e.numpy(), self.sig2bs.numpy(), [], []
        if self.mode == 'spatial':
            return self.sig2e.numpy(), np.concatenate([self.sig2bs.numpy(), self.lengthscale]), [], []
        if self.mode == 'glmm':
            return None, self.sig2bs.numpy(), [], []
        if hasattr(self, 'rhos') and len(self.est_cors) > 0:
            return self.sig2e.numpy(), self.sig2bs.numpy(), self.rhos.numpy(), []
        else:
            return self.sig2e.numpy(), self.sig2bs.numpy(), [], []

    def get_table(self, Z_idx):
        Z_unique, _ = tf.unique(Z_idx)
        Z_mapto = tf.range(tf.shape(Z_unique)[0], dtype=tf.int64)
        table = tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    Z_unique,
                    Z_mapto,
                    key_dtype=tf.int64,
                    value_dtype=tf.int64,
                ),
                num_oov_buckets=1,
            )
        return table
    
    def get_indices(self, N, Z_idx, min_Z):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx - min_Z], axis=1)

    def get_indices_v1(self, N, Z_idx):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx], axis=1)

    def getZ(self, N, Z_idx, min_Z, max_Z):
        if self.Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices(N, Z_idx, min_Z)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, max_Z - min_Z + 1)))
    
    def getZ_v1(self, N, Z_idx):
        if self.Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices_v1(N, Z_idx)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, tf.reduce_max(Z_idx) + 1)))

    def getD(self, min_Z, max_Z):
        a = tf.range(min_Z, max_Z + 1)
        d = tf.shape(a)[0]
        ix_ = tf.reshape(tf.stack([tf.repeat(a, d), tf.tile(a, [d])], 1), [d, d, 2])
        M = tf.gather_nd(self.dist_matrix, ix_)
        M = tf.cast(M, tf.float32)
        D = self.sig2bs[0] * tf.math.exp(-M / (2 * self.lengthscale))
        return D
    
    def getG(self, min_Z, max_Z):
        a = tf.range(min_Z, max_Z + 1)
        d = tf.shape(a)[0]
        ix_ = tf.reshape(tf.stack([tf.repeat(a, d), tf.tile(a, [d])], 1), [d, d, 2])
        M = tf.gather_nd(self.dist_matrix, ix_)
        M = tf.cast(M, tf.float32)
        G = self.sig2bs[0] * M
        return G
    
    def inverse_gaussian_cdf(self, u):
        return tf.math.erfinv(2 * tf.clip_by_value(u, 1e-5, 1 - 1e-5) - 1) * np.sqrt(2)
    
    def custom_loss_lm(self, y_true, y_pred, Z_idxs):
        n_int = K.shape(y_true)[0]
        n_float = K.cast(K.shape(y_true)[0], tf.float32)
        V, resid, sig2, sd_sqrt_V = self.mode.V_batch(y_true, y_pred, Z_idxs, self.Z_non_linear, n_int,
                                                      self.sig2e, self.sig2bs, self.rhos, self.est_cors,
                                                      self.dist_matrix, self.lengthscale)
        u = self.distribution.cdf_batch(resid, sig2)
        m = self.inverse_gaussian_cdf(u)
        sum_log_pdf = self.distribution.sum_log_pdf_batch(resid, sig2, n_float)
        if self.Z_non_linear:
            V_inv = tf.linalg.inv(V)
            V_inv_m = K.dot(V_inv, m)
        else:
            V_inv_m = tf.linalg.solve(V, m)
        logdet = self.mode.logdet(V, sd_sqrt_V)
        mV_invm = K.dot(K.transpose(m), V_inv_m)
        mtm = K.dot(K.transpose(m), m)
        total_loss = 0.5 * logdet + 0.5 * mV_invm - 0.5 * mtm + 0.5 * sum_log_pdf
        return total_loss

    def custom_loss_glm(self, y_true, y_pred, Z_idxs):
        Z_idx = K.squeeze(Z_idxs[0], axis=1)
        a, _ = tf.unique(Z_idx)
        i_sum = tf.zeros(shape=(1,1))
        for i in a:
            y_i = y_true[Z_idx == i]
            f_i = y_pred[Z_idx == i]
            yf = K.dot(K.transpose(y_i), f_i)
            k_sum = tf.zeros(shape=(1,1))
            for k in range(self.nGQ):
                sqrt2_sigb_xk = np.sqrt(2) * tf.sqrt(self.sig2bs[0]) * self.x_ks[k]
                y_sum_x = K.sum(y_i) * sqrt2_sigb_xk
                log_gamma_sum = K.sum(K.log(1 + K.exp(f_i + sqrt2_sigb_xk)))
                k_sum = k_sum + K.exp(yf + y_sum_x - log_gamma_sum) * self.w_ks[k] / np.sqrt(np.pi)
            i_sum = i_sum + K.log(k_sum)
        return -i_sum
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, y_true, y_pred, Z_idxs):
        if self.mode == 'glmm':
            self.add_loss(self.custom_loss_glm(y_true, y_pred, Z_idxs))
        else:
            self.add_loss(self.custom_loss_lm(y_true, y_pred, Z_idxs))
        return y_pred
