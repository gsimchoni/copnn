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
    
    def bivariate_normal_cdf(self, x, mean, cov_matrix):
        def scipy_cdf(x_np):
            # Compute the CDF using SciPy
            return np.array([stats.multivariate_normal.cdf(point, mean=mean, cov=cov_matrix) for point in x_np], dtype=np.float32)
        # def scipy_cdf(x_np):
        #     # Compute the CDF using SciPy
        #     return stats.multivariate_normal.cdf(x_np, mean=mean, cov=cov_matrix)
        
        # Use tf.py_function to wrap the SciPy CDF function
        # cdf_values = tf.py_function(func=scipy_cdf, inp=[x], Tout=tf.float32)
        # return cdf_values
        return tf.numpy_function(scipy_cdf, [x], tf.float32)
    
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

    def f(self, y_pred_0, y_pred_1, y_true_0, y_true_1):
        if y_true_0 == 0.0 and y_true_1 == 0.0:
            return y_pred_0 + y_pred_1 + 1.0
        elif y_true_0 == 1.0 and y_true_1 == 1.0:
            return y_pred_0 + y_pred_1 + 2.0
        elif y_true_0 == 0.0 and y_true_1 == 1.0:
            return y_pred_0 + y_pred_1 + 3.0
        elif y_true_0 == 1.0 and y_true_1 == 0.0:
            return y_pred_0 + y_pred_1 + 4.0
        else:
            return 0.0
    
    def custom_loss_glm1(self, y_true, y_pred, Z_idxs):
        batch_size = tf.shape(y_true)[0]

        def pairwise_op(i, j):
            y_true_0 = y_true[i]
            y_true_1 = y_true[j]
            y_pred_0 = y_pred[i]
            y_pred_1 = y_pred[j]
            return self.f(y_pred_0, y_pred_1, y_true_0, y_true_1)
        
        # Create indices for all pairs
        indices = tf.where(tf.not_equal(tf.range(batch_size)[:, None], tf.range(batch_size)[None, :]))
        i_indices, j_indices = indices[:, 0], indices[:, 1]

        # Compute the custom pairwise function for each pair
        pairwise_values = tf.map_fn(lambda idx: pairwise_op(idx[0], idx[1]), (i_indices, j_indices), dtype=tf.float32)

        # Average over all pairs
        loss = tf.reduce_mean(pairwise_values)
        return loss
    
    def custom_loss_glm(self, y_true, y_pred, Z_idxs):
        # y_true and y_pred are both (batch_size, n_features) tensors
        batch_size = tf.shape(y_true)[0]

        # make y_pred probabilities
        p_pred = tf.math.exp(y_pred) / (1 + tf.math.exp(y_pred))

        z_pred = self.inverse_gaussian_cdf(p_pred)

        # Expand dimensions to compute pairwise operations
        y_true_expanded_1 = tf.expand_dims(y_true, axis=0)  # Shape: (1, batch_size, n_features)
        y_true_expanded_2 = tf.expand_dims(y_true, axis=1)  # Shape: (batch_size, 1, n_features)
        z_pred_expanded_1 = tf.expand_dims(z_pred, axis=0)  # Shape: (1, batch_size, n_features)
        z_pred_expanded_2 = tf.expand_dims(z_pred, axis=1)  # Shape: (batch_size, 1, n_features)
        tf.print(z_pred_expanded_1.shape)
        tf.print(z_pred_expanded_2.shape)

        # Calculate pairwise quantities based on the values of y_true
        condition_00 = (y_true_expanded_1 == 0) & (y_true_expanded_2 == 0)
        condition_11 = (y_true_expanded_1 == 1) & (y_true_expanded_2 == 1)
        condition_01 = (y_true_expanded_1 == 0) & (y_true_expanded_2 == 1)
        condition_10 = (y_true_expanded_1 == 1) & (y_true_expanded_2 == 0)

        # Initialize pairwise quantity tensor
        pairwise_quantity = tf.zeros_like(z_pred_expanded_1 + z_pred_expanded_2)

        # Case 1: Both y_true are 0
        mean = [0.0, 0.0]
        # cov_matrix = tf.constant([[1.0, self.sig2bs[0]/2], [self.sig2bs[0]/2, 1.0]])
        # cov_matrix = K.constant([[1.0, 0.5], [0.5, 1.0]])
        cov_matrix = [[1.0, 0.5], [0.5, 1.0]]
        pairwise_quantity += tf.where(condition_00, self.bivariate_normal_cdf(z_pred_expanded_1, mean, cov_matrix), 0)
        # pairwise_quantity += tf.where(condition_00, self.bivariate_normal_cdf([[0.3, 0.5], [0.8, -0.8]], mean, cov_matrix), 0)

        # Case 2: Both y_true are 1
        pairwise_quantity += tf.where(condition_11, z_pred_expanded_1 + z_pred_expanded_2 + 2, 0)

        # Case 3: First y_true is 0 and second y_true is 1
        pairwise_quantity += tf.where(condition_01, z_pred_expanded_1 + z_pred_expanded_2 + 3, 0)

        # Case 4: First y_true is 1 and second y_true is 0
        pairwise_quantity += tf.where(condition_10, z_pred_expanded_1 + z_pred_expanded_2 + 4, 0)

        # Sum over features
        pairwise_sum = K.sum(pairwise_quantity, axis=-1)  # Shape: (batch_size, batch_size)

        # Average over all pairs (excluding self-pairs)
        loss = K.sum(pairwise_sum) / (tf.cast(batch_size, tf.float32) * (tf.cast(batch_size, tf.float32) - 1))

        return loss
    
    def bivariate_normal_cdf(self, x, y, mean, cov_matrix):
        # This function computes the CDF of a bivariate normal distribution for given x, y
        def cdf_approximation(point):
            # tf.print(point)
            # Example approximation for the CDF, this should be replaced with the actual CDF calculation
            mean_vec = tf.constant(mean, dtype=tf.float32)
            point_centered = point - mean_vec
            # Ensure correct shapes for matrix multiplication
            point_centered = tf.reshape(point_centered, (-1, 2))
            # tf.print('**', point_centered.shape)
            cov_inv = tf.linalg.inv(cov_matrix)
            cdf_value = tf.reduce_sum(point)#tf.exp(-0.5 * tf.reduce_sum(tf.multiply(point_centered, tf.linalg.matvec(cov_inv, point_centered)), axis=-1))
            # tf.print(cdf_value)
            return cdf_value

        points = tf.stack([x, y], axis=-1)
        # tf.print(points.shape)
        cdf_values = tf.map_fn(cdf_approximation, points, dtype=tf.float32)
        return cdf_values
    
    def tf_normal_cdf(self, x):
            """CDF of the standard normal distribution using TensorFlow"""
            return 0.5 * (tf.math.erf(x / tf.sqrt(2.0)) + 1.0)
    
    def bivariate_normal_cdf2(self, x, y, mean, r):
        def scipy_cdf(x_np):
            # Compute the CDF using SciPy
            return stats.multivariate_normal.cdf(x_np, mean=mean, cov=[[1.0, r], [r, 1.0]]).astype(np.float32)
            # return np.array([stats.multivariate_normal.cdf(point, mean=mean, cov=cov_matrix) for point in x_np], dtype=np.float32)
        
            # Use tf.py_function to wrap the SciPy CDF function
        def cdf_approximation1(point):
            return tf.numpy_function(scipy_cdf, [point], tf.float32)
            # return tf.py_function(func=scipy_cdf, inp=[point], Tout=tf.float32)
        
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

        # points = tf.stack([x, y], axis=-1)
        # cdf_values = tf.map_fn(cdf_approximation2, points, dtype=tf.float32)
        # cdf_values = cdf_approximation(x, y, r)

        points = tf.stack([x, y], axis=1)
        cdf_values = tf.map_fn(cdf_approximation, points, dtype=tf.float32)
        return cdf_values

    def custom_f(self, p_pred_i, p_pred_j, y_true_i, y_true_j):
        # Define the parameters for the bivariate normal distribution
        mean = [0.0, 0.0]
        cov_matrix = [[1.0, 0.5], [0.5, 1.0]] #tf.constant([[1.0, self.sig2bs[0]/2], [self.sig2bs[0]/2, 1.0]], dtype=tf.float32)
        
        # Compute the CDF for the pair (y_pred_i, y_pred_j)
        cdf_value = self.bivariate_normal_cdf2(self.inverse_gaussian_cdf(1 - p_pred_i), self.inverse_gaussian_cdf(1 - p_pred_j), mean, 0.5)

            # Apply conditions based on y_true values
        condition_00 = tf.logical_and(tf.equal(y_true_i, 0), tf.equal(y_true_j, 0))
        condition_01 = tf.logical_and(tf.equal(y_true_i, 0), tf.equal(y_true_j, 1))
        condition_10 = tf.logical_and(tf.equal(y_true_i, 1), tf.equal(y_true_j, 0))
        condition_11 = tf.logical_and(tf.equal(y_true_i, 1), tf.equal(y_true_j, 1))
        
        # Initialize result
        result = tf.zeros_like(cdf_value)

        # Apply conditions
        result = tf.where(condition_00, cdf_value, result)
        result = tf.where(condition_01, 1 - p_pred_i - cdf_value, result)
        result = tf.where(condition_10, 1 - p_pred_j - cdf_value, result)
        result = tf.where(condition_11, p_pred_i + p_pred_j + cdf_value - 1, result)
        
        return result
    
    def custom_loss_glm2(self, y_true, y_pred, Z_idxs):
        # y_true and y_pred are both (batch_size, n_features) tensors
        batch_size = tf.shape(y_true)[0]

        # make y_pred probabilities
        # p_pred = 1 - tf.math.exp(y_pred) / (1 + tf.math.exp(y_pred))
        p_pred = self.tf_normal_cdf(y_pred)

        # z_pred = self.inverse_gaussian_cdf(1 - p_pred)

        # Expand dimensions to compute pairwise operations
        y_true_expanded_1 = tf.expand_dims(y_true, axis=0)  # Shape: (1, batch_size, n_features)
        y_true_expanded_2 = tf.expand_dims(y_true, axis=1)  # Shape: (batch_size, 1, n_features)
        p_pred_expanded_1 = tf.expand_dims(p_pred, axis=0)  # Shape: (1, batch_size, n_features)
        p_pred_expanded_2 = tf.expand_dims(p_pred, axis=1)  # Shape: (batch_size, 1, n_features)

        # Broadcast the expanded tensors to pairwise shapes
        y_true_expanded_1 = tf.broadcast_to(y_true_expanded_1, [tf.shape(y_true)[0], tf.shape(y_true)[0], tf.shape(y_true)[1]])
        y_true_expanded_2 = tf.broadcast_to(y_true_expanded_2, [tf.shape(y_true)[0], tf.shape(y_true)[0], tf.shape(y_true)[1]])
        p_pred_expanded_1 = tf.broadcast_to(p_pred_expanded_1, [tf.shape(y_pred)[0], tf.shape(y_pred)[0], tf.shape(y_pred)[1]])
        p_pred_expanded_2 = tf.broadcast_to(p_pred_expanded_2, [tf.shape(y_pred)[0], tf.shape(y_pred)[0], tf.shape(y_pred)[1]])

        # Apply the custom function for all pairs
        pairwise_loss = tf.vectorized_map(
            lambda pair: self.custom_f(pair[0][0], pair[1][0], pair[2][0], pair[3][0]), 
            (p_pred_expanded_1, p_pred_expanded_2, y_true_expanded_1, y_true_expanded_2)
        )
        
        # Sum over pairs and average the loss
        total_loss = tf.reduce_sum(pairwise_loss)
        num_pairs = tf.shape(y_true)[0] * (tf.shape(y_true)[0] - 1)  # batch_size * (batch_size - 1)
        
        # Avoid division by zero
        total_loss = tf.cond(num_pairs > 0, lambda: total_loss / tf.cast(num_pairs, tf.float32), lambda: total_loss)

        return total_loss
    
    def custom_loss_glm3(self, y_true, y_pred, Z_idxs):

        # make y_pred probabilities
        # p_pred = 1 - tf.math.exp(y_pred) / (1 + tf.math.exp(y_pred))
        p_pred = self.tf_normal_cdf(y_pred)
        z_pred = self.inverse_gaussian_cdf(1 - p_pred)

        def compute_pairwise_ll1(i, j):
            p_i = tf.gather(p_pred, i)
            p_j = tf.gather(p_pred, j)
            inv_cdf_i = tf.gather(z_pred, i)
            inv_cdf_j = tf.gather(z_pred, j)
            y_i = tf.gather(y_true, i)
            y_j = tf.gather(y_true, j)
            
            # inv_cdf_i = self.inverse_gaussian_cdf(1 - p_i)
            # inv_cdf_j = tf_inverse_gaussian_cdf(1 - p_j)
            
            C = self.bivariate_normal_cdf2(inv_cdf_i, inv_cdf_j, [0.0, 0.0], self.sig2bs[0] / (self.sig2bs[0] + self.sig2e))

            pl = tf.case([
                (tf.logical_and(tf.equal(y_i, 0), tf.equal(y_j, 0)), lambda: C),
                (tf.logical_and(tf.equal(y_i, 0), tf.equal(y_j, 1)), lambda: 1 - p_i - C),
                (tf.logical_and(tf.equal(y_i, 1), tf.equal(y_j, 0)), lambda: 1 - p_j - C),
                (tf.logical_and(tf.equal(y_i, 1), tf.equal(y_j, 1)), lambda: p_i + p_j + C - 1)
            ], exclusive=True)
            
            ll = tf.reshape(tf.math.log(tf.maximum(pl, self.tol)), (1, 1))
            # tf.print(ll.shape)
            return ll
        
        def compute_pairwise_ll(y_i, y_j, p_i, p_j, inv_cdf_i, inv_cdf_j):
            # inv_cdf_i = tf_inverse_gaussian_cdf(1 - p_i)
            # inv_cdf_j = tf_inverse_gaussian_cdf(1 - p_j)
            
            C = self.bivariate_normal_cdf2(inv_cdf_i, inv_cdf_j, [0.0, 0.0], self.sig2bs[0] / (self.sig2bs[0] + self.sig2e))
            
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

        Z_idx = K.squeeze(Z_idxs[0], axis=1)
        unique_groups = tf.unique(Z_idx)[0]
        total_loss = tf.zeros(shape=(1,1))
        # total_loss = tf.constant([0.0], dtype=tf.float32)

        for group in unique_groups:
            mask = tf.equal(Z_idx, group)
            indices = tf.where(mask)[:, 0]
            n = tf.shape(indices)[0]
            if n < 2:
                continue  # Skip groups with less than 2 elements

            # Create pairwise indices
            i_idx, j_idx = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
            mask = i_idx < j_idx

            i_idx = tf.boolean_mask(i_idx, mask)
            j_idx = tf.boolean_mask(j_idx, mask)

            i_indices = tf.gather(indices, i_idx)
            j_indices = tf.gather(indices, j_idx)

            y_i = tf.gather(y_true, i_indices)
            y_j = tf.gather(y_true, j_indices)
            p_i = tf.gather(p_pred, i_indices)
            p_j = tf.gather(p_pred, j_indices)
            z_i = tf.gather(z_pred, i_indices)
            z_j = tf.gather(z_pred, j_indices)

            total_loss_g = compute_pairwise_ll(y_i, y_j, p_i, p_j, z_i, z_j)
            # total_loss_g = tf.zeros(shape=(1,1))
            # for i in range(n):
            #     for j in range(i + 1, n):
            #         total_loss_g -= compute_pairwise_ll(indices[i], indices[j])
            
            # Avoid division by zero
            num_pairs = n * (n - 1) / 2
            total_loss_g = tf.reduce_sum(total_loss_g)
            total_loss_g = tf.cond(num_pairs > 0, lambda: total_loss_g / tf.cast(num_pairs, tf.float32), lambda: total_loss_g)
            total_loss -= total_loss_g

        return total_loss
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, y_true, y_pred, Z_idxs):
        if self.y_type == 'binary':
            self.add_loss(self.custom_loss_glm3(y_true, y_pred, Z_idxs))
        else:
            self.add_loss(self.custom_loss_lm(y_true, y_pred, Z_idxs))
        return y_pred
