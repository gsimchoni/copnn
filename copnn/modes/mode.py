import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import sparse, stats
from sklearn.model_selection import train_test_split

from copnn.utils import copulize, RegData

class Mode:
    def __init__(self, mode_par):
        self.mode_par = mode_par
        self.tol = 1e-5
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.mode_par == other
        else:
            return NotImplemented
    
    def __str__(self):
        return self.mode_par
    
    def get_indices(self, N, Z_idx, min_Z):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx - min_Z], axis=1)
    
    def getZ_batch(self, N, Z_idx, min_Z, max_Z, Z_non_linear):
        if Z_non_linear:
            return Z_idx
        Z_idx = K.squeeze(Z_idx, axis=1)
        indices = self.get_indices(N, Z_idx, min_Z)
        return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, max_Z - min_Z + 1)))
    
    def getD_batch(self, min_Z, max_Z, dist_matrix, lengthscale, sig2bs):
        a = tf.range(min_Z, max_Z + 1)
        d = tf.shape(a)[0]
        ix_ = tf.reshape(tf.stack([tf.repeat(a, d), tf.tile(a, [d])], 1), [d, d, 2])
        M = tf.gather_nd(dist_matrix, ix_)
        M = tf.cast(M, tf.float32)
        D = sig2bs[0] * tf.math.exp(-M / (2 * lengthscale))
        return D

    def get_D(self):
        raise NotImplementedError('The get_D method is not implemented.')
    
    def sample_conditional_b_hat(self, z_samp, distribution, sig2, y_min):
        if distribution == 'exponential':
            # allow for inputing y_min as shift MLE for SExp distribution
            b_hat = (distribution.quantile(np.clip(stats.norm.cdf(z_samp),0, 1-1e-16), x_min=0) * np.sqrt(sig2))
            b_hat += y_min
        else:
            b_hat = (distribution.quantile(np.clip(stats.norm.cdf(z_samp),0, 1-1e-16)) * np.sqrt(sig2))
        return b_hat

    def sample_fe(self, params, N):
        n_fixed_effects = params['n_fixed_effects']
        X = np.random.uniform(-1, 1, N * n_fixed_effects).reshape((N, n_fixed_effects))
        betas = np.ones(n_fixed_effects)
        Xbeta = params['fixed_intercept'] + X @ betas
        if params['X_non_linear']:
            fX = Xbeta * np.cos(Xbeta) + 2 * X[:, 0] * X[:, 1]
        else:
            fX = Xbeta
        fX = fX / fX.std()
        return X, fX

    def sample_re(self):
        Zb, sig2, Z_idx_list, t, coords, dist_matrix = None, None, None, None, None, None
        time_fe = 0
        return Zb, sig2, Z_idx_list, t, time_fe, coords, dist_matrix

    def create_df(self, X, y, Z_idx_list, t, coords):
        time2measure_dict = None
        df = pd.DataFrame(X)
        x_cols = ['X' + str(i) for i in range(X.shape[1])]
        df.columns = x_cols
        for k, Z_idx in enumerate(Z_idx_list):
            df['z' + str(k)] = Z_idx
        df['y'] = y
        return df, x_cols, time2measure_dict
    
    def train_test_split(self, df, test_size, pred_unknown_clusters, cluster_q, pred_future):
        if pred_unknown_clusters:
            train_clusters, test_clusters = train_test_split(range(cluster_q), test_size=test_size)
            X_train = df[df['z0'].isin(train_clusters)]
            X_test = df[df['z0'].isin(test_clusters)]
            y_train = df['y'][df['z0'].isin(train_clusters)]
            y_test = df['y'][df['z0'].isin(test_clusters)]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop('y', axis=1), df['y'], test_size=test_size, shuffle=not pred_future)
        return X_train, X_test, y_train, y_test
    
    def V_batch(self, y_true, y_pred, Z_idxs, Z_non_linear, n_int, sig2e, sig2bs, rhos, est_cors, dist_matrix, lengthscale):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def logdet(self, V, sd_sqrt_V):
        sgn, logdet = tf.linalg.slogdet(V)
        logdet = sgn * logdet
        return logdet
    
    def predict_re_continuous(self):
        raise NotImplementedError('The predict_re_continuous method is not implemented.')
    
    def predict_re_binary(self):
        raise NotImplementedError('The predict_re_binary method is not implemented.')
    
    def predict_re(self, y_type, X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                   Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train=10000):
        if y_type == 'continuous':
            return self.predict_re_continuous(X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                                              Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train)
        elif y_type == 'binary':
            return self.predict_re_binary(X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
                                          Z_non_linear, model, ls, rhos, est_cors, dist_matrix, distribution, sample_n_train)
        else:
            raise ValueError(y_type + ' is an unknown y_type')
    
    def get_Zb_hat(self, model, X_test, Z_non_linear, qs, b_hat, n_sig2bs, y_type, is_blup=False):
        Zb_hat = b_hat[X_test['z0']]
        return Zb_hat
    
    def build_net_input(self, x_cols, X_train, qs, n_sig2bs, n_sig2bs_spatial):
        raise NotImplementedError('The build_net_input method is not implemented.')
    
    def build_Z_nll_inputs(self, Z_inputs, Z_non_linear, qs, Z_embed_dim_pct):
        Z_nll_inputs = Z_inputs
        ls = None
        return Z_nll_inputs, ls
    
    def sample_conditional_RE(self, b, D_inv, j):
        # Extract components for conditional distribution
        omega_jj = D_inv[j, j]
        omega_j_minus_j = D_inv[j, np.arange(len(D_inv)) != j]
        
        # Conditional mean
        conditional_mean = -omega_j_minus_j @ b[np.arange(len(D_inv)) != j] / omega_jj
        
        # Conditional variance
        conditional_variance = 1 / omega_jj
        
        # Sample from the conditional distribution
        b_j_sample = np.random.normal(conditional_mean, np.sqrt(conditional_variance))
        
        return b_j_sample
    
    def posterior_j(self, b, j, Z, y_train, y_pred_tr):
        latent_L = y_pred_tr + Z @ b + np.random.normal(0, 1, size=y_pred_tr.shape)
        L_j = latent_L[Z[:, j].nonzero()[0]]
        Y_j = y_train[Z[:, j].nonzero()[0]]
        Phi_j = np.clip(stats.norm.cdf(L_j), self.tol, 1 - self.tol)
        return np.exp(np.sum(Y_j * np.log(Phi_j) + (1 - Y_j) * np.log(1 - Phi_j)))
    
    def metropolis_hastings(self, y_true, y_pred, Z, D_inv):
        total_q = D_inv.shape[0]
        b_current = np.random.randn(total_q)
        
        # Metropolis-Hastings parameters
        n_iter = 100
        b_samples = np.zeros((n_iter, total_q))
        burn_in = 0.2

        # Metropolis-Hastings algorithm
        for i in range(n_iter):
            if (i + 1) % (n_iter // 10) == 0 or i == n_iter - 1:
                progress = int((i + 1) / n_iter * 100)
                print(f"MH Progress: {progress}%", end='\r')
            
            # Propose new values for b_j
            b_proposal = b_current.copy()
            
            for j in range(total_q):
                b_proposal[j] = self.sample_conditional_RE(b_proposal, D_inv, j)
                
                # Calculate the posterior for the current and proposed b_j
                posterior_current = self.posterior_j(b_current, j, Z, y_true, y_pred)
                posterior_proposal = self.posterior_j(b_proposal, j, Z, y_true, y_pred)
                
                # Acceptance ratio
                acceptance_ratio = posterior_proposal / posterior_current
                
                # Accept or reject the proposal
                if np.random.rand() < acceptance_ratio or (posterior_current == 0 and posterior_proposal > 0):
                    b_current[j] = b_proposal[j]
                
            b_samples[i, :] = b_current
        print('\n')

        # Calculate mean of samples
        b_hat = np.mean(b_samples[int(burn_in * n_iter):, :], axis=0)
        return b_hat


def generate_data(mode, y_type, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos,
                  distribution, test_size, pred_unknown_clusters, params):
    X, fX = mode.sample_fe(params, N)
    e = np.random.normal(0, np.sqrt(sig2e), N)
    Zb, time_fe, sig2, Z_idx_list, t, coords, dist_matrix = mode.sample_re(params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos)
    z = (Zb + e)/np.sqrt(sig2)
    if y_type == 'continuous':
        b_cop = copulize(z, distribution, sig2)
    elif y_type == 'binary':
        b_cop = Zb + e
    y = fX + time_fe + b_cop
    if y_type == 'binary':
        probit = True
        if not probit:
            p = np.exp(y)/(1 + np.exp(y))
            y = np.random.binomial(1, p, size=N)
        else:
            y[y > 0.0] = 1.0
            y[y <= 0.0] = 0.0
    df, x_cols, time2measure_dict = mode.create_df(X, y, Z_idx_list, t, coords)
    X_train, X_test, y_train, y_test = mode.train_test_split(df, test_size, pred_unknown_clusters, params, qs, q_spatial)
    return RegData(X_train, X_test, y_train, y_test, x_cols, dist_matrix, time2measure_dict, b_cop)
