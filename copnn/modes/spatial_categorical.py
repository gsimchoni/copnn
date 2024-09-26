import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import sparse, stats
from scipy.spatial.distance import pdist, squareform
from tensorflow.keras.layers import Embedding, Input, Reshape

from copnn.modes.mode import Mode
from copnn.utils import get_dummies, sample_ns


class SpatialCategorical(Mode):
    def __init__(self):
        super().__init__('spatial_categorical')
    
    def get_D_categorical(self, qs, sig2bs):
        D_hat = sparse.eye(np.sum(qs))
        D_hat.setdiag(np.repeat(sig2bs, qs))
        return D_hat
    
    def get_D_spatial(self, sig2bs_spatial, dist_matrix):
        return sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
    
    def sample_re(self, params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos):
        _, _, _, t, time_fe, _, _ = super().sample_re()
        # spatial RE
        coords = np.stack([np.random.uniform(-10, 10, q_spatial), np.random.uniform(-10, 10, q_spatial)], axis=1)
        # ind = np.lexsort((coords[:, 1], coords[:, 0]))    
        # coords = coords[ind]
        dist_matrix = squareform(pdist(coords)) ** 2
        D = self.get_D_spatial(sig2bs_spatial, dist_matrix)
        b = np.random.multivariate_normal(np.zeros(q_spatial), D, 1)[0]
        ns = sample_ns(N, q_spatial, params['n_per_cat'])
        Z_idx = np.repeat(range(q_spatial), ns)
        gZb = np.repeat(b, ns)
        sig2 = sig2e + sig2bs_spatial[0]

        # categrical RE
        sum_gZbs = gZb
        Z_idx_list = [Z_idx]
        for k, q in enumerate(qs):
            ns = sample_ns(N, q, params['n_per_cat'])
            Z_idx = np.repeat(range(q), ns)
            Z_idx_list.append(Z_idx)
            if params['Z_non_linear']:
                Z = get_dummies(Z_idx, q)
                l = int(q * params['Z_embed_dim_pct'] / 100.0)
                b = np.random.normal(0, np.sqrt(sig2bs[k]), l)
                W = np.random.uniform(-1, 1, q * l).reshape((q, l))
                if params.get('Z_non_linear_embed', False):
                    if q <= 200:
                        ZW = (Z.toarray()[:,None,:]*W.T[None,:,:]*np.cos(Z.toarray()[:,None,:]*W.T[None,:,:])).sum(axis=2)
                    else:
                        zw_list = []
                        for i in range(Z.shape[0]):
                            zw_list.append(Z[i, :] * W * np.cos(Z[i, :] * W))
                        ZW = np.concatenate(zw_list, axis=0)
                else:
                    ZW = Z @ W
                gZb = ZW @ b
            else:
                b = np.random.normal(0, np.sqrt(sig2bs[k]), q)
                gZb = np.repeat(b, ns)
            sum_gZbs += gZb
        sig2 += np.sum(sig2bs)
        return sum_gZbs, time_fe, sig2, Z_idx_list, t, coords, dist_matrix
    
    def create_df(self, X, y, Z_idx_list, t, coords):
        df, x_cols, time2measure_dict = super().create_df(X, y, Z_idx_list, t, coords)
        coords_df = pd.DataFrame(coords[Z_idx_list[0]])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
        df = pd.concat([df, coords_df], axis=1)
        x_cols.extend(co_cols)
        return df, x_cols, time2measure_dict
    
    def train_test_split(self, df, test_size, pred_unknown_clusters, params, qs, q_spatial):
        return super().train_test_split(df, test_size, pred_unknown_clusters, q_spatial, False)
    
    def V_batch(self, y_true, y_pred, Z_idxs, Z_non_linear, n_int, sig2e, sig2bs, rhos, est_cors, dist_matrix, lengthscale):
        sd_sqrt_V = None
        V = sig2e * tf.eye(n_int)
        categoricals_loc = 1
        for k, Z_idx in enumerate(Z_idxs[categoricals_loc:]):
            min_Z = tf.reduce_min(Z_idx)
            max_Z = tf.reduce_max(Z_idx)
            Z = self.getZ_batch(n_int, Z_idx, min_Z, max_Z, Z_non_linear)
            sig2bs_loc = k + 2 # first 2 sig2bs go to kernel
            V += sig2bs[sig2bs_loc] * K.dot(Z, K.transpose(Z))

        min_Z = tf.reduce_min(Z_idxs[0])
        max_Z = tf.reduce_max(Z_idxs[0])
        D = self.getD_batch(min_Z, max_Z, dist_matrix, lengthscale, sig2bs)
        Z = self.getZ_batch(n_int, Z_idxs[0], min_Z, max_Z, Z_non_linear)
        V += K.dot(Z, K.dot(D, K.transpose(Z)))
        sig2 = K.sum(sig2bs) + sig2e
        V /= sig2
        resid = y_true - y_pred
        return V, resid, sig2, sd_sqrt_V
    
    def predict_re_binary(self, X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                   Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train=10000):
        gZ_train_categ, _, n_cats, samp = self.get_Z_matrices_categorical(X_train, X_test, qs, sig2bs, Z_non_linear, model, ls, sample_n_train)
        gZ_train_spat = self.get_Z_matrices_spatial2(X_train, q_spatial, samp)
        D_categ = self.get_D_categorical(n_cats, sig2bs)
        D_spat = self.get_D_spatial(sig2bs_spatial, dist_matrix)
        gZ_train = sparse.hstack([gZ_train_categ, gZ_train_spat])
        D = sparse.block_diag((D_categ, D_spat))
        D_inv = np.linalg.inv(D)
        b_hat = self.metropolis_hastings(y_train.values, y_pred_tr, gZ_train, D_inv)
        return b_hat
    
    def predict_re_continuous(self, X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                   Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train=10000):
        gZ_train_categ, gZ_test_categ, n_cats, samp = self.get_Z_matrices_categorical(X_train, X_test, qs, sig2bs, Z_non_linear, model, ls, sample_n_train)
        D_categ = self.get_D_categorical(n_cats, sig2bs)
        V = gZ_train_categ @ D_categ @ gZ_train_categ.T + sparse.eye(gZ_train_categ.shape[0]) * sig2e
        V_te = gZ_test_categ @ D_categ @ gZ_test_categ.T + sparse.eye(gZ_test_categ.shape[0]) * sig2e
        gZ_train_spat = self.get_Z_matrices_spatial2(X_train, q_spatial, samp)
        gZ_test_spat = self.get_Z_matrices_spatial2(X_test, q_spatial)
        D_spat = self.get_D_spatial(sig2bs_spatial, dist_matrix)
        V += gZ_train_spat @ D_spat @ gZ_train_spat.T
        V_te += gZ_test_spat @ D_spat @ gZ_test_spat.T
        gZ_train = sparse.hstack([gZ_train_spat, gZ_train_categ])
        gZ_test = sparse.hstack([gZ_test_spat, gZ_test_categ])
        D = sparse.block_diag((D_spat, D_categ))
        D /= (sig2bs.sum() + sig2bs_spatial[0] + sig2e)
        V /= (sig2bs.sum() + sig2bs_spatial[0] + sig2e)
        V_te /= (sig2bs.sum() + sig2bs_spatial[0] + sig2e)
        y_standardized = (y_train.values[samp] - y_pred_tr[samp])/np.sqrt(sig2bs.sum() + sig2bs_spatial[0] + sig2e)
        y_min = (y_train.values[samp] - y_pred_tr[samp]).min()
        V_inv_y = np.linalg.solve(V, stats.norm.ppf(np.clip(distribution.cdf(y_standardized), 0 + 1e-16, 1 - 1e-16)))
        b_hat_mean = gZ_test @ D @ gZ_train.T @ V_inv_y
        if gZ_test.shape[0] > 10000:
            b_hat = self.sample_conditional_b_hat(b_hat_mean, distribution, sig2bs.sum() + sig2bs_spatial[0] + sig2e, y_min)
        else:
            if D.shape[0] < V.shape[0]:
                D_inv = sparse.linalg.inv(D.tocsc())
                sig2e_rho = sig2e / (sig2bs.sum() + sig2bs_spatial[0] + sig2e)
                A = gZ_train.T @ gZ_train / sig2e_rho + D_inv
                V_inv = sparse.eye(V.shape[0]) / sig2e_rho - (1/(sig2e_rho**2)) * gZ_train @ sparse.linalg.inv(A) @ gZ_train.T
            else:
                V_inv = np.linalg.inv(V)
            sig2 = sig2bs.sum() + sig2bs_spatial[0] + sig2e
            A = gZ_test_categ @ (D_categ / sig2) @ gZ_train_categ.T + gZ_test_spat @ (D_spat / sig2) @ gZ_train_spat.T
            b_hat_cov = V_te - A @ V_inv @ A.T
            z_samp = stats.multivariate_normal.rvs(mean = b_hat_mean, cov = b_hat_cov, size = 1000)
            b_hat_array = self.sample_conditional_b_hat(z_samp, distribution, sig2bs.sum() + sig2bs_spatial[0] + sig2e, y_min)
            b_hat = b_hat_array.mean(axis=0)
        return b_hat

    def get_Zb_hat(self, model, X_test, Z_non_linear, qs, q_spatial, b_hat, n_sig2bs, y_type, is_blup=False):
        return b_hat
    
    def get_Z_matrices_categorical(self, X_train, X_test, qs, sig2bs, Z_non_linear, model, ls, sample_n_train):
        gZ_trains = []
        gZ_tests = []
        delta_loc = 1
        for k in range(len(sig2bs)):
            gZ_train = get_dummies(X_train['z' + str(k + delta_loc)].values, qs[k])
            gZ_test = get_dummies(X_test['z' + str(k + delta_loc)].values, qs[k])
            if Z_non_linear:
                W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                gZ_train = gZ_train @ W_est
                gZ_test = gZ_test @ W_est
            gZ_trains.append(gZ_train)
            gZ_tests.append(gZ_test)
        if Z_non_linear:
            if X_train.shape[0] > 10000:
                samp = np.random.choice(X_train.shape[0], 10000, replace=False)
            else:
                samp = np.arange(X_train.shape[0])
            gZ_train = np.hstack(gZ_trains)
            gZ_train = gZ_train[samp]
            gZ_test = np.hstack(gZ_tests)
            n_cats = ls
        else:
            gZ_train = sparse.hstack(gZ_trains)
            gZ_test = sparse.hstack(gZ_tests)
            n_cats = qs
            samp = np.arange(X_train.shape[0])
            # in spatial_and_categoricals increase this as you can
            if X_train.shape[0] > sample_n_train:
                samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
            elif X_train.shape[0] > 100000:
                # Z linear, multiple categoricals, V is relatively sparse, will solve with sparse.linalg.cg
                # consider sampling or "inducing points" approach if matrix is huge
                # samp = np.random.choice(X_train.shape[0], 100000, replace=False)
                pass
            gZ_train = gZ_train.tocsr()[samp]
            gZ_test = gZ_test.tocsr()
        return gZ_train, gZ_test, n_cats, samp
    
    def get_Z_matrices_spatial2(self, X_train, q_spatial, samp=None):
        gZ_train = get_dummies(X_train['z0'].values, q_spatial)
        if samp is not None:
            gZ_train = gZ_train[samp]
        return gZ_train

    def get_Z_matrices_spatial(self, X_train, q_spatial, sample_n_train):
        gZ_train = get_dummies(X_train['z0'].values, q_spatial)
        # increase this as you can
        if X_train.shape[0] > sample_n_train:
            samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        return gZ_train, samp
      
    def build_net_input(self, x_cols, X_train, qs, n_sig2bs, n_sig2bs_spatial):
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
        X_input = Input(shape=(X_train[x_cols].shape[1],))
        y_true_input = Input(shape=(1,))
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        n_sig2bs_init = n_sig2bs_spatial + len(qs)
        n_RE_inputs = 1 + len(qs)
        for _ in range(n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
        return X_input, y_true_input, Z_inputs, x_cols, z_cols, n_sig2bs_init

    def build_Z_nll_inputs(self, Z_inputs, Z_non_linear, qs, Z_embed_dim_pct):
        if Z_non_linear:
            Z_nll_inputs = []
            ls = []
            for k, q in enumerate(qs):
                l = int(q * Z_embed_dim_pct / 100.0)
                Z_embed = Embedding(q, l, input_length=1, name='Z_embed' + str(k))(Z_inputs[k])
                Z_embed = Reshape(target_shape=(l, ))(Z_embed)
                Z_nll_inputs.append(Z_embed)
                ls.append(l)
        else:
            Z_nll_inputs, ls = super().build_Z_nll_inputs(Z_inputs, Z_non_linear, qs, Z_embed_dim_pct)
        return Z_nll_inputs, ls