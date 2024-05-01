import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import sparse, stats
from tensorflow.keras.layers import Embedding, Input, Reshape

from copnn.modes.mode import Mode
from copnn.utils import get_dummies, sample_ns


class Categorical(Mode):
    def __init__(self):
        super().__init__('categorical')
    
    def sample_re(self, params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos):
        _, _, _, t, time_fe, coords, dist_matrix = super().sample_re()
        sum_gZbs = 0
        Z_idx_list = []
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
        sig2 = sig2e + np.sum(sig2bs)
        return sum_gZbs, time_fe, sig2, Z_idx_list, t, coords, dist_matrix
    
    def train_test_split(self, df, test_size, pred_unknown_clusters, params, qs, q_spatial):
        return super().train_test_split(df, test_size, pred_unknown_clusters, qs[0], False)
    
    def V_batch(self, y_true, y_pred, Z_idxs, Z_non_linear, n_int, sig2e, sig2bs, rhos, est_cors, dist_matrix, lengthscale):
        sd_sqrt_V = None
        V = sig2e * tf.eye(n_int)
        for k, Z_idx in enumerate(Z_idxs):
            min_Z = tf.reduce_min(Z_idx)
            max_Z = tf.reduce_max(Z_idx)
            Z = self.getZ_batch(n_int, Z_idx, min_Z, max_Z, Z_non_linear)
            V += sig2bs[k] * K.dot(Z, K.transpose(Z))
        sig2 = K.sum(sig2bs) + sig2e
        V /= sig2
        resid = y_true - y_pred
        return V, resid, sig2, sd_sqrt_V
    
    def predict_re(self, X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                   Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train=10000):
        gZ_trains = []
        gZ_tests = []
        for k in range(len(sig2bs)):
            gZ_train = get_dummies(X_train['z' + str(k)].values, qs[k])
            gZ_test = get_dummies(X_test['z' + str(k)].values, qs[k])
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
            if X_train.shape[0] > sample_n_train and False:
                samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
            elif X_train.shape[0] > 100000:
                # Z linear, multiple categoricals, V is relatively sparse, will solve with sparse.linalg.cg
                # consider sampling or "inducing points" approach if matrix is huge
                # samp = np.random.choice(X_train.shape[0], 100000, replace=False)
                pass
            gZ_train = gZ_train.tocsr()[samp]
            gZ_test = gZ_test.tocsr()
        D = self.get_D_est(n_cats, sig2bs)
        V = gZ_train @ D @ gZ_train.T + sparse.eye(gZ_train.shape[0]) * sig2e
        V_te = gZ_test @ D @ gZ_test.T + sparse.eye(gZ_test.shape[0]) * sig2e
        V /= (np.sum(sig2bs) + sig2e)
        V_te /= (np.sum(sig2bs) + sig2e)
        D /= (np.sum(sig2bs) + sig2e)
        y_standardized = (y_train.values[samp] - y_pred_tr[samp])/np.sqrt(np.sum(sig2bs) + sig2e)
        y_min = (y_train.values[samp] - y_pred_tr[samp]).min()
        if Z_non_linear:
            V_inv_y = np.linalg.solve(V, stats.norm.ppf(np.clip(distribution.cdf(y_standardized), 0 + 1e-16, 1 - 1e-16)))
        else:
            V_inv_y = sparse.linalg.cg(V, stats.norm.ppf(np.clip(distribution.cdf(y_standardized), 0 + 1e-16, 1 - 1e-16)))[0]
        # woodbury
        D_inv = self.get_D_est(n_cats, (np.sum(sig2bs) + sig2e)/sig2bs)
        sig2e_rho = sig2e / (np.sum(sig2bs) + sig2e)
        A = gZ_train.T @ gZ_train / sig2e_rho + D_inv
        V_inv = sparse.eye(V.shape[0]) / sig2e_rho - (1/(sig2e_rho**2)) * gZ_train @ sparse.linalg.inv(A) @ gZ_train.T
        if len(qs) > 1:
            b_hat_mean = gZ_test @ D @ gZ_train.T @ V_inv_y
            b_hat_cov = V_te - gZ_test @ D @ gZ_train.T @ V_inv @ gZ_train @ D @ gZ_test.T
        else:
            b_hat_mean = D @ gZ_train.T @ V_inv_y
            b_hat_cov = sparse.eye(D.shape[0]) - D @ gZ_train.T @ V_inv @ gZ_train @ D
            # Omega_m = D * (np.sum(sig2bs) + sig2e) + sparse.eye(D.shape[0]) * sig2e
            # Omega_m /= (np.sum(sig2bs) + sig2e)
        z_samp = stats.multivariate_normal.rvs(mean = b_hat_mean, cov = b_hat_cov.toarray(), size = 10000)
        b_hat = self.sample_conditional_b_hat(z_samp, distribution, np.sum(sig2bs) + sig2e, y_min)
        return b_hat
    
    def get_Zb_hat(self, model, X_test, Z_non_linear, qs, b_hat, n_sig2bs, is_blup=False):
        if is_blup:
            if Z_non_linear or len(qs) > 1:
                Z_tests = []
                for k, q in enumerate(qs):
                    Z_test = get_dummies(X_test['z' + str(k)], q)
                    if Z_non_linear:
                        W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                        Z_test = Z_test @ W_est
                    Z_tests.append(Z_test)
                if Z_non_linear:
                    Z_test = np.hstack(Z_tests)
                else:
                    Z_test = sparse.hstack(Z_tests)
                Zb_hat = Z_test @ b_hat
            else:
                Zb_hat = super().get_Zb_hat(model, X_test, Z_non_linear, qs, b_hat, n_sig2bs)
        elif len(qs) > 1:
            Zb_hat = b_hat
        else:
            Zb_hat = super().get_Zb_hat(model, X_test, Z_non_linear, qs, b_hat, n_sig2bs)
        return Zb_hat
    
    def build_net_input(self, x_cols, X_train, qs, n_sig2bs, n_sig2bs_spatial):
        X_input = Input(shape=(X_train[x_cols].shape[1],))
        y_true_input = Input(shape=(1,))
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        n_sig2bs_init = len(qs)
        n_RE_inputs = len(qs)
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
