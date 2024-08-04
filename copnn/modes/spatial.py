import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from tensorflow.keras.layers import Input

from copnn.modes.mode import Mode
from copnn.utils import get_dummies, sample_ns


class Spatial(Mode):
    def __init__(self):
        super().__init__('spatial')
    
    def get_D(self, sig2bs_spatial, dist_matrix):
        return sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
    
    def sample_re(self, params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos):
        _, _, _, t, time_fe, _, _ = super().sample_re()
        coords = np.stack([np.random.uniform(-10, 10, q_spatial), np.random.uniform(-10, 10, q_spatial)], axis=1)
        # ind = np.lexsort((coords[:, 1], coords[:, 0]))    
        # coords = coords[ind]
        dist_matrix = squareform(pdist(coords)) ** 2
        D = self.get_D(sig2bs_spatial, dist_matrix)
        b = np.random.multivariate_normal(np.zeros(q_spatial), D, 1)[0]
        ns = sample_ns(N, q_spatial, params['n_per_cat'])
        Z_idx = np.repeat(range(q_spatial), ns)
        gZb = np.repeat(b, ns)
        sig2 = sig2e + sig2bs_spatial[0]
        return gZb, time_fe, sig2, [Z_idx], t, coords, dist_matrix
    
    def create_df(self, X, y, Z_idx_list, t, coords):
        df, x_cols, time2measure_dict = super().create_df(X, y, Z_idx_list, t, coords)
        coords_df = pd.DataFrame(coords[Z_idx_list[0]])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
        df = pd.concat([df, coords_df], axis=1)
        return df, x_cols, time2measure_dict
    
    def train_test_split(self, df, test_size, pred_unknown_clusters, params, qs, q_spatial):
        return super().train_test_split(df, test_size, pred_unknown_clusters, q_spatial, False)
    
    def V_batch(self, y_true, y_pred, Z_idxs, Z_non_linear, n_int, sig2e, sig2bs, rhos, est_cors, dist_matrix, lengthscale):
        sd_sqrt_V = None
        V = sig2e * tf.eye(n_int)
        min_Z = tf.reduce_min(Z_idxs[0])
        max_Z = tf.reduce_max(Z_idxs[0])
        D = self.getD_batch(min_Z, max_Z, dist_matrix, lengthscale, sig2bs)
        Z = self.getZ_batch(n_int, Z_idxs[0], min_Z, max_Z, Z_non_linear)
        V += K.dot(Z, K.dot(D, K.transpose(Z)))
        sig2 = sig2bs[0] + sig2e
        V /= sig2
        resid = y_true - y_pred
        return V, resid, sig2, sd_sqrt_V
    
    def predict_re(self, X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                   Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train=10000):
        gZ_train = get_dummies(X_train['z0'].values, q_spatial)
        D = self.get_D(sig2bs_spatial, dist_matrix)
        # increase this as you can
        if X_train.shape[0] > sample_n_train:
            samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        V = gZ_train @ D @ gZ_train.T + np.eye(gZ_train.shape[0]) * sig2e
        V /= (sig2bs_spatial[0] + sig2e)
        D /= (sig2bs_spatial[0] + sig2e)
        y_standardized = (y_train.values[samp] - y_pred_tr[samp])/np.sqrt(sig2bs_spatial[0] + sig2e)
        y_min = (y_train.values[samp] - y_pred_tr[samp]).min()
        V_inv_y = np.linalg.solve(V, stats.norm.ppf(np.clip(distribution.cdf(y_standardized), 0 + 1e-16, 1 - 1e-16)))
        b_hat_mean = D @ gZ_train.T @ V_inv_y
        # b_hat = distribution.quantile(stats.norm.cdf(b_hat_mean)) * np.sqrt(sig2bs_spatial[0] + sig2e)
        D_inv = np.linalg.inv(D)
        sig2e_rho = sig2e / (sig2bs_spatial[0] + sig2e)
        A = gZ_train.T @ gZ_train / sig2e_rho + D_inv
        V_inv = np.eye(V.shape[0]) / sig2e_rho - (1/(sig2e_rho**2)) * gZ_train @ np.linalg.inv(A) @ gZ_train.T
        Omega_m = D * (sig2bs_spatial[0] + sig2e) + np.eye(D.shape[0]) * sig2e
        Omega_m /= (sig2bs_spatial[0] + sig2e)
        b_hat_cov = Omega_m - D @ gZ_train.T @ V_inv @ gZ_train @ D
        z_samp = stats.multivariate_normal.rvs(mean = b_hat_mean, cov = b_hat_cov, size = 10000)
        b_hat_array = self.sample_conditional_b_hat(z_samp, distribution, sig2bs_spatial[0] + sig2e, y_min)
        b_hat = b_hat_array.mean(axis=0)
        return b_hat
    
    def build_net_input(self, x_cols, X_train, qs, n_sig2bs, n_sig2bs_spatial):
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
        X_input = Input(shape=(X_train[x_cols].shape[1],))
        y_true_input = Input(shape=(1,))
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        n_sig2bs_init = 1
        Z_inputs = [Input(shape=(1,), dtype=tf.int64)]
        return X_input, y_true_input, Z_inputs, x_cols, z_cols, n_sig2bs_init
