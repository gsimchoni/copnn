import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import sparse, stats
from tensorflow.keras.layers import Input

from copnn.modes.mode import Mode
from copnn.utils import get_dummies, sample_ns, get_cov_mat

class Longitudinal(Mode):
    def __init__(self):
        super().__init__('longitudinal')
    
    def sample_re(self, params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos):
        _, _, _, _, _, coords, dist_matrix = super().sample_re()
        ns = sample_ns(N, qs[0], params['n_per_cat'])
        Z_idx = np.repeat(range(qs[0]), ns)
        max_period = np.arange(ns.max())
        t = np.concatenate([max_period[:k] for k in ns]) / max_period[-1]
        estimated_cors = [] if params['estimated_cors'] is None else params['estimated_cors']
        cov_mat = get_cov_mat(sig2bs, rhos, estimated_cors)
        D = sparse.kron(cov_mat, sparse.eye(qs[0]))
        bs = np.random.multivariate_normal(np.zeros(len(sig2bs)), cov_mat, qs[0])
        b = bs.reshape((qs[0] * len(sig2bs),), order = 'F')
        Z0 = sparse.csr_matrix(get_dummies(Z_idx, qs[0]))
        Z_list = [Z0]
        time_fe = 0
        for k in range(1, len(sig2bs)):
            time_fe += t ** k # fixed part t + t^2 + t^3 + ...
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Z = sparse.hstack(Z_list)
        Zb = Z @ b
        V_diagonal = (Z @ D @ Z.T + sparse.eye(N) * sig2e).diagonal()
        return Zb, time_fe, V_diagonal, [Z_idx], t, coords, dist_matrix
    
    def create_df(self, X, y, Z_idx_list, t, coords):
        df, x_cols, time2measure_dict = super().create_df(X, y, Z_idx_list, t, coords)
        df['t'] = t
        x_cols.append('t')
        time2measure_dict = {t: i for i, t in enumerate(np.sort(df['t'].unique()))}
        return df, x_cols, time2measure_dict
    
    def train_test_split(self, df, test_size, pred_unknown_clusters, params, qs, q_spatial):
        pred_future = params.get('longitudinal_predict_future', False)
        if  pred_future:
            # test set is "the future" or those obs with largest t
            df.sort_values('t', inplace=True)
        return super().train_test_split(df, test_size, pred_unknown_clusters, qs[0], pred_future)
    
    def V_batch(self, y_true, y_pred, Z_idxs, Z_non_linear, n_int, sig2e, sig2bs, rhos, est_cors, dist_matrix, lengthscale):
        V = sig2e * tf.eye(n_int)
        min_Z = tf.reduce_min(Z_idxs[0])
        max_Z = tf.reduce_max(Z_idxs[0])
        Z0 = self.getZ_batch(n_int, Z_idxs[0], min_Z, max_Z, Z_non_linear)
        Z_list = [Z0]
        for k in range(1, len(sig2bs)):
            T = tf.linalg.tensor_diag(K.squeeze(Z_idxs[1], axis=1) ** k)
            Z = K.dot(T, Z0)
            Z_list.append(Z)
        for k in range(len(sig2bs)):
            for j in range(len(sig2bs)):
                if k == j:
                    sig = sig2bs[k] 
                else:
                    rho_symbol = ''.join(map(str, sorted([k, j])))
                    if rho_symbol in est_cors:
                        rho = rhos[est_cors.index(rho_symbol)]
                        sig = rho * tf.math.sqrt(sig2bs[k]) * tf.math.sqrt(sig2bs[j])
                    else:
                        continue
                V += sig * K.dot(Z_list[j], K.transpose(Z_list[k]))
        sd_sqrt_V = tf.math.sqrt(tf.linalg.tensor_diag_part(V))
        S = tf.linalg.tensor_diag(1/sd_sqrt_V)
        V = S @ V @ S
        sd_sqrt_V = tf.expand_dims(sd_sqrt_V, -1)
        resid = (y_true - y_pred)/sd_sqrt_V
        sig2 = tf.constant(1.0)
        return V, resid, sig2, sd_sqrt_V
    
    def logdet(self, V, sd_sqrt_V):
        logdet = super().logdet(V, sd_sqrt_V)
        logdet += 2 * tf.math.log(tf.reduce_prod(sd_sqrt_V))
        return logdet
    
    def predict_re(self, X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                   Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train=10000):
        q = qs[0]
        Z0 = get_dummies(X_train['z0'], q)
        Z0_te = get_dummies(X_test['z0'], q)
        t = X_train['t'].values
        t_te = X_test['t'].values
        N = X_train.shape[0]
        N_te = X_test.shape[0]
        Z_list = [Z0]
        Z_list_te = [Z0_te]
        for k in range(1, len(sig2bs)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
            Z_list_te.append(sparse.spdiags(t_te ** k, 0, N_te, N_te) @ Z0_te)
        gZ_train = sparse.hstack(Z_list)
        gZ_test = sparse.hstack(Z_list_te)
        cov_mat = get_cov_mat(sig2bs, rhos, est_cors)
        D = sparse.kron(cov_mat, sparse.eye(q))
        V = gZ_train @ D @ gZ_train.T + sparse.eye(gZ_train.shape[0]) * sig2e
        V_diagonal = V.diagonal()
        sd_sqrt_V = sparse.diags(1/np.sqrt(V_diagonal))
        V = sd_sqrt_V @ V @ sd_sqrt_V
        V_te = gZ_test @ D @ gZ_test.T + sparse.eye(gZ_test.shape[0]) * sig2e
        V_diagonal_te = V_te.diagonal()
        sd_sqrt_V_te = sparse.diags(1/np.sqrt(V_diagonal_te))
        y_standardized = (y_train.values - y_pred_tr)/np.sqrt(V_diagonal)
        y_min = (y_train.values - y_pred_tr).min()
        V_inv_y = sparse.linalg.cg(V, stats.norm.ppf(np.clip(distribution.cdf(y_standardized), 0 + 1e-16, 1 - 1e-16)))[0]
        b_hat = D @ gZ_train.T @ sd_sqrt_V @ V_inv_y
        # b_hat = distribution.quantile(stats.norm.cdf(b_hat)) * np.sqrt(V_diagonal)
        D_inv = sparse.linalg.inv(D.tocsc())
        sig2e_inv = sparse.diags(V_diagonal / sig2e)
        A = gZ_train.T @ sd_sqrt_V @ sig2e_inv @ sd_sqrt_V @ gZ_train + D_inv
        V_inv = sig2e_inv - sig2e_inv @ sd_sqrt_V @ gZ_train @ sparse.linalg.inv(A) @ gZ_train.T @ sd_sqrt_V @ sig2e_inv
        if gZ_test.shape[0] <= 10000:
            b_hat_mean = sd_sqrt_V_te @ gZ_test @ b_hat
            Omega_m = sd_sqrt_V_te @ V_te @ sd_sqrt_V_te
            b_hat_cov = Omega_m - sd_sqrt_V_te @ gZ_test @ D @ gZ_train.T @ sd_sqrt_V @ V_inv @ sd_sqrt_V @ gZ_train @ D @ gZ_test.T @ sd_sqrt_V_te
            z_samp = stats.multivariate_normal.rvs(mean = b_hat_mean, cov = b_hat_cov.toarray(), size = 10000)
            b_hat = self.sample_conditional_b_hat(z_samp, distribution, b_hat_mean, b_hat_cov.toarray(), 1.0, y_min) * np.sqrt(V_diagonal_te)
        else:
            # does not seem correct
            b_hat_mean = b_hat
            b_hat_cov = sparse.eye(D.shape[0]) - D @ gZ_train.T @ V_inv @ gZ_train @ D / ((np.sum(sig2bs) + sig2e)**2)
            z_samp = stats.multivariate_normal.rvs(mean = b_hat_mean, cov = b_hat_cov.toarray(), size = 10000)
            b_hat = self.sample_conditional_b_hat(z_samp, distribution, b_hat_mean, b_hat_cov.toarray(), 1.0, y_min)
            b_hat = gZ_test @ b_hat * np.sqrt(V_diagonal_te)
        # b_hat_cov = sparse.eye(gZ_test.shape[0]) - sd_sqrt_V_te @ gZ_test @ D @ gZ_train.T @ sd_sqrt_V @ V_inv @ sd_sqrt_V @ gZ_train @ D @ gZ_test.T @ sd_sqrt_V_te
        return b_hat
    
    def get_Zb_hat(self, model, X_test, Z_non_linear, qs, b_hat, n_sig2bs, is_blup=False):
        if is_blup:
            q = qs[0]
            Z0 = get_dummies(X_test['z0'], q)
            t = X_test['t'].values
            N = X_test.shape[0]
            Z_list = [Z0]
            for k in range(1, n_sig2bs):
                Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
            Z_test = sparse.hstack(Z_list)
            Zb_hat = Z_test @ b_hat
        else:
            Zb_hat = b_hat
        return Zb_hat
    
    def build_net_input(self, x_cols, X_train, qs, n_sig2bs, n_sig2bs_spatial):
        X_input = Input(shape=(X_train[x_cols].shape[1],))
        y_true_input = Input(shape=(1,))
        z_cols = ['z0', 't']
        n_sig2bs_init = n_sig2bs
        Z_input = Input(shape=(1,), dtype=tf.int64)
        t_input = Input(shape=(1,))
        Z_inputs = [Z_input, t_input]
        return X_input, y_true_input, Z_inputs, x_cols, z_cols, n_sig2bs_init