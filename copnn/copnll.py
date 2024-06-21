from os import name
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
from scipy import special, stats
import tensorflow.keras.backend as K


class COPNLL(Layer):
    """COPNN Negative Log Likelihood Loss Layer"""

    def __init__(self, mode, y_type, sig2e, sig2bs, rhos = [], est_cors = [], Z_non_linear=False, dist_matrix=None, lengthscale=None, distribution=None):
        super(COPNLL, self).__init__(dynamic=False)
        self.sig2bs = tf.Variable(
            sig2bs, name='sig2bs', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
        self.Z_non_linear = Z_non_linear
        self.mode = mode
        self.y_type = y_type
        self.rhos = None
        self.est_cors = None
        self.dist_matrix = None
        self.lengthscale = None
        if self.mode in ['categorical', 'longitudinal', 'spatial', 'spatial_embedded', 'spatial_and_categoricals', 'mme'] and self.y_type == 'continuous':
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
        if self.y_type == 'binary':
            self.sig2e = tf.Variable(
                sig2e, name='sig2e', constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
            self.nGQ = 5
            self.x_ks, self.w_ks = np.polynomial.hermite.hermgauss(self.nGQ)
        self.distribution = distribution
        self.tol = 1e-200

    def get_vars(self):
        if self.mode in ['categorical', 'spatial_embedded', 'spatial_and_categoricals', 'mme'] and self.y_type == 'continuous':
            return self.sig2e.numpy(), self.sig2bs.numpy(), [], []
        if self.mode == 'spatial':
            return self.sig2e.numpy(), np.concatenate([self.sig2bs.numpy(), self.lengthscale]), [], []
        if self.y_type == 'binary':
            return self.sig2e.numpy(), self.sig2bs.numpy(), [], []
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

    def tf_normal_cdf(self, x):
            """CDF of the standard normal distribution using TensorFlow"""
            return 0.5 * (tf.math.erf(x / tf.sqrt(2.0)) + 1.0)
    
    def bivariate_normal_cdf(self, x, y, mean, r):
        def scipy_cdf(x_np):
            return stats.multivariate_normal.cdf(x_np, mean=mean, cov=[[1.0, r], [r, 1.0]]).astype(np.float32)
        
        def cdf_approximation1(point):
            return tf.numpy_function(scipy_cdf, [point], tf.float32)
        
        def cdf_approximation(point):
            """Approximate the CDF of a bivariate normal distribution"""
            dh, dk = point[0], point[1]
            
            # Flip signs
            dh = tf.cast(-dh, dtype=tf.float32)
            dk = tf.cast(-dk, dtype=tf.float32)

            inf = tf.constant(float('inf'), dtype=tf.float32)

            if tf.equal(dh, inf) or tf.equal(dk, inf): 
                return tf.constant(0.0, dtype=tf.float32)
            
            if tf.equal(dh, -inf):
                if tf.equal(dk, -inf):
                    return tf.constant(1.0, dtype=tf.float32)
                else:
                    return self.tf_normal_cdf(-dk)
            if tf.equal(dk, -inf):
                return self.tf_normal_cdf(-dh)

            if tf.equal(r, 0.0):
                return self.tf_normal_cdf(-dh) * self.tf_normal_cdf(-dk)
            
            tp = tf.constant(2 * 3.141592653589793, dtype=tf.float32)
            h = dh
            k = dk
            hk = h * k
            bvn = tf.constant(0.0, dtype=tf.float32)

            def get_w_x(abs_r):
                if abs_r < 0.3:
                    w = tf.constant([0.1713244923791705, 0.3607615730481384, 0.4679139345726904], dtype=tf.float32)
                    x = tf.constant([0.9324695142031522, 0.6612093864662647, 0.238619186083197], dtype=tf.float32)
                elif abs_r < 0.75:
                    w = tf.constant([0.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659,
                                    0.2334925365383547, 0.2491470458134029], dtype=tf.float32)
                    x = tf.constant([0.9815606342467191, 0.904117256370475, 0.769902674194305, 0.5873179542866171,
                                    0.3678314989981802, 0.1252334085114692], dtype=tf.float32)
                else:
                    w = tf.constant([0.01761400713915212, 0.04060142980038694, 0.06267204833410905, 0.08327674157670475,
                                    0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
                                    0.1491729864726037, 0.1527533871307259], dtype=tf.float32)
                    x = tf.constant([0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188,
                                    0.7463319064601508, 0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
                                    0.2277858511416451, 0.07652652113349733], dtype=tf.float32)
                return w, x

            w, x = get_w_x(tf.abs(r))
            w = tf.concat([w, w], axis=0)
            x = tf.concat([1 - x, 1 + x], axis=0)

            if tf.abs(r) < 0.925:
                hs = (h * h + k * k) / 2
                asr = tf.asin(r) / 2
                sn = tf.sin(asr * x)
                bvn = tf.reduce_sum(tf.exp((sn * hk - hs) / (1 - sn * sn)) * w)
                bvn = bvn * asr / tp + self.tf_normal_cdf(-h) * self.tf_normal_cdf(-k)
            else:
                if r < 0:
                    k = -k
                    hk = -hk

                if tf.abs(r) < 1:
                    as1 = 1 - r * r
                    a = tf.sqrt(as1)
                    bs = (h - k) * (h - k)
                    asr = - (bs / as1 + hk) / 2
                    c = (4 - hk) / 8
                    d = (12 - hk) / 80

                    if asr > -100:
                        bvn = a * tf.exp(asr) * (1 - c * (bs - as1) * (1 - d * bs) / 3 + c * d * as1 * as1)

                    if hk > -100:
                        b = tf.sqrt(bs)
                        sp = tf.sqrt(tp) * self.tf_normal_cdf(-b / a)
                        bvn = bvn - tf.exp(-hk / 2) * sp * b * (1 - c * bs * (1 - d * bs) / 3)

                    a = a / 2
                    xs = (a * x) * (a * x)
                    asr = - (bs / xs + hk) / 2
                    ix = asr > -100
                    xs = tf.boolean_mask(xs, ix)
                    sp = 1 + c * xs * (1 + 5 * d * xs)
                    rs = tf.sqrt(1 - xs)
                    ep = tf.exp(- (hk / 2) * xs / (1 + rs) / (1 + rs)) / rs
                    bvn = (a * tf.reduce_sum(tf.boolean_mask(tf.exp(asr[ix]) * (sp - ep), ix) * w[ix]) - bvn) / tp

                if r > 0:
                    bvn = bvn + self.tf_normal_cdf(-tf.maximum(h, k))
                else:
                    if h >= k:
                        bvn = -bvn
                    else:
                        if h < 0:
                            L = self.tf_normal_cdf(k) - self.tf_normal_cdf(h)
                        else:
                            L = self.tf_normal_cdf(-h) - self.tf_normal_cdf(-k)
                        bvn = L - bvn

            return tf.maximum(0.0, tf.minimum(1.0, bvn))

        points = tf.stack([x, y], axis=1)
        cdf_values = tf.map_fn(cdf_approximation, points)
        return cdf_values
    
    def custom_loss_glm(self, y_true, y_pred, Z_idxs):
        
        def compute_pairwise_ll(y_i, y_j, p_i, p_j, inv_cdf_i, inv_cdf_j):
            C = self.bivariate_normal_cdf(inv_cdf_i, inv_cdf_j, [0.0, 0.0], self.sig2bs[0] / (self.sig2bs[0] + self.sig2e))
            
            loss_00 = C
            loss_01 = 1 - p_i - C
            loss_10 = 1 - p_j - C
            loss_11 = p_i + p_j + C - 1
            
            pl = tf.where(tf.logical_and(tf.equal(y_i, 0), tf.equal(y_j, 0)), loss_00,
                            tf.where(tf.logical_and(tf.equal(y_i, 0), tf.equal(y_j, 1)), loss_01,
                            tf.where(tf.logical_and(tf.equal(y_i, 1), tf.equal(y_j, 0)), loss_10,
                                    loss_11)))
            
            ll = tf.math.log(tf.maximum(pl, self.tol))
            return ll

        p_pred = self.tf_normal_cdf(y_pred)
        inv_cdf_pred = self.inverse_gaussian_cdf(1 - p_pred)
        Z_idx = K.squeeze(Z_idxs[0], axis=1)
        n = tf.shape(y_true)[0]
        i_idx, j_idx = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        mask = i_idx < j_idx

        i_idx = tf.boolean_mask(i_idx, mask)
        j_idx = tf.boolean_mask(j_idx, mask)

        y_i = tf.gather(y_true, i_idx)
        y_j = tf.gather(y_true, j_idx)
        p_i = tf.gather(p_pred, i_idx)
        p_j = tf.gather(p_pred, j_idx)
        inv_cdf_i = tf.gather(inv_cdf_pred, i_idx)
        inv_cdf_j = tf.gather(inv_cdf_pred, j_idx)
        Z_i = tf.gather(Z_idx, i_idx)
        Z_j = tf.gather(Z_idx, j_idx)

        same_group_mask = tf.equal(Z_i, Z_j)

        y_i = tf.boolean_mask(y_i, same_group_mask)
        y_j = tf.boolean_mask(y_j, same_group_mask)
        p_i = tf.boolean_mask(p_i, same_group_mask)
        p_j = tf.boolean_mask(p_j, same_group_mask)
        inv_cdf_i = tf.boolean_mask(inv_cdf_i, same_group_mask)
        inv_cdf_j = tf.boolean_mask(inv_cdf_j, same_group_mask)

        pair_ll = compute_pairwise_ll(y_i, y_j, p_i, p_j, inv_cdf_i, inv_cdf_j)
        total_nll = -tf.reduce_sum(pair_ll)
        num_pairs = tf.shape(y_i)[0]
        total_nll = tf.cond(num_pairs > 0, lambda: total_nll / tf.cast(num_pairs, tf.float32), lambda: total_nll)
        return total_nll
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, y_true, y_pred, Z_idxs):
        if self.y_type == 'binary':
            self.add_loss(self.custom_loss_glm(y_true, y_pred, Z_idxs))
        else:
            self.add_loss(self.custom_loss_lm(y_true, y_pred, Z_idxs))
        return y_pred
