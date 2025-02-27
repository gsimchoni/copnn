import numpy as np
import pandas as pd
from scipy import sparse, stats, special

from tensorflow.keras import Model

from copnn.utils import get_cov_mat, get_dummies


def get_D_est(qs, sig2bs):
    D_hat = sparse.eye(np.sum(qs))
    D_hat.setdiag(np.repeat(sig2bs, qs))
    return D_hat

def conditional_b_hat(distribution, b_hat_mean, b_hat_cov, n_te, sig2):
    b_hat = []
    for i in range(n_te):
        b_hat_norm_quantiles = stats.norm.ppf(np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), loc=b_hat_mean[i], scale=b_hat_cov[i,i])
        b_hat_orig_quantiles = distribution.quantile(stats.norm.cdf(b_hat_norm_quantiles)) * np.sqrt(sig2)
        b_hat_i = 0.28871*b_hat_orig_quantiles[3] + 0.18584*(b_hat_orig_quantiles[2] + b_hat_orig_quantiles[4]) + 0.13394*(b_hat_orig_quantiles[1] + b_hat_orig_quantiles[5]) + 0.036128*(b_hat_orig_quantiles[0] + b_hat_orig_quantiles[6])
        # b_hat_i = -0.3039798 * b_hat_orig_quantiles[2] + 1.3039798 * b_hat_orig_quantiles[3]
        b_hat.append(b_hat_i)
    b_hat = np.array(b_hat)
    return b_hat

def sample_conditional_b_hat(distribution, b_hat_mean, b_hat_cov, sig2, n=10000):
    q_samp = stats.multivariate_normal.rvs(mean = b_hat_mean, cov = b_hat_cov, size = n)
    b_hat = (distribution.quantile(np.clip(stats.norm.cdf(q_samp),0, 1-1e-16)) * np.sqrt(sig2)).mean(axis=0)
    return b_hat

def calc_b_hat(X_train, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
    Z_non_linear, model, ls, mode, rhos, est_cors, dist_matrix, weibull_ests, y_type, sample_n_train=10000):
    experimental = False
    if y_type == 'binary':
        sig2e = 1.0
    if mode in ['categorical', 'spatial_categorical'] and y_type == 'continuous':
        if Z_non_linear or len(qs) > 1 or mode == 'spatial_categorical':
            delta_loc = 0
            if mode == 'spatial_categorical':
                delta_loc = 1
            gZ_trains = []
            for k in range(len(sig2bs)):
                gZ_train = get_dummies(X_train['z' + str(k + delta_loc)].values, qs[k])
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    gZ_train = gZ_train @ W_est
                gZ_trains.append(gZ_train)
            if Z_non_linear:
                if X_train.shape[0] > 10000:
                    samp = np.random.choice(X_train.shape[0], 10000, replace=False)
                else:
                    samp = np.arange(X_train.shape[0])
                gZ_train = np.hstack(gZ_trains)
                gZ_train = gZ_train[samp]
                n_cats = ls
            else:
                gZ_train = sparse.hstack(gZ_trains)
                n_cats = qs
                samp = np.arange(X_train.shape[0])
                if not experimental:
                    # in spatial_categorical increase this as you can
                    if mode == 'spatial_categorical' and X_train.shape[0] > sample_n_train:
                        samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
                    elif X_train.shape[0] > 100000:
                        # Z linear, multiple categoricals, V is relatively sparse, will solve with sparse.linalg.cg
                        # consider sampling or "inducing points" approach if matrix is huge
                        # samp = np.random.choice(X_train.shape[0], 100000, replace=False)
                        pass
                gZ_train = gZ_train.tocsr()[samp]
            if not experimental:
                D = get_D_est(n_cats, sig2bs)
                V = gZ_train @ D @ gZ_train.T + sparse.eye(gZ_train.shape[0]) * sig2e
                if mode == 'spatial_categorical':
                    gZ_train_spatial = get_dummies(X_train['z0'].values, q_spatial)
                    D_spatial = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
                    gZ_train_spatial = gZ_train_spatial[samp]
                    V += gZ_train_spatial @ D_spatial @ gZ_train_spatial.T
                    gZ_train = sparse.hstack([gZ_train, gZ_train_spatial])
                    D = sparse.block_diag((D, D_spatial))
                    V_inv_y = np.linalg.solve(V, y_train.values[samp] - y_pred_tr[samp])
                else:
                    if Z_non_linear:
                        V_inv_y = np.linalg.solve(V, (y_train.values[samp] - y_pred_tr[samp]))
                    else:
                        V_inv_y = sparse.linalg.cg(V, (y_train.values[samp] - y_pred_tr[samp]))[0]
                b_hat = D @ gZ_train.T @ V_inv_y
            else:
                if mode == 'spatial_categorical':
                    raise ValueError('experimental inverse not yet implemented in this mode')
                D_inv = get_D_est(n_cats, 1 / sig2bs)
                A = gZ_train.T @ gZ_train / sig2e + D_inv
                b_hat = np.linalg.inv(A.toarray()) @ gZ_train.T / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
                b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
        else:
            b_hat = single_random_intercept_b_hat(X_train, y_train, y_pred_tr, qs, sig2e, sig2bs)
    elif mode == 'longitudinal' and y_type == 'continuous':
        q = qs[0]
        Z0 = get_dummies(X_train['z0'], q)
        t = X_train['t'].values
        N = X_train.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2bs)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        gZ_train = sparse.hstack(Z_list)
        cov_mat = get_cov_mat(sig2bs, rhos, est_cors)
        D = sparse.kron(cov_mat, sparse.eye(q))
        V = gZ_train @ D @ gZ_train.T + sparse.eye(gZ_train.shape[0]) * sig2e
        V_inv_y = sparse.linalg.cg(V, y_train.values - y_pred_tr)[0]
        b_hat = D @ gZ_train.T @ V_inv_y
    elif y_type == 'binary':
        nGQ = 5
        x_ks, w_ks = np.polynomial.hermite.hermgauss(nGQ)
        a = np.unique(X_train['z0'])
        b_hat_numerators = []
        b_hat_denominators = []
        q = q_spatial if mode == 'spatial' else qs[0]
        b_hat0 = sig2bs_spatial[0] if mode == 'spatial' else sig2bs[0]
        for i in range(q):
            if i in a:
                i_vec = X_train['z0'] == i
                y_i = y_train.values[i_vec]
                f_i = y_pred_tr[i_vec]
                yf = np.dot(y_i, f_i)
                k_sum_num = 0
                k_sum_den = 0
                for k in range(nGQ):
                    sqrt2_sigb_xk = np.sqrt(2) * np.sqrt(b_hat0) * x_ks[k]
                    y_sum_x = y_i.sum() * sqrt2_sigb_xk
                    log_gamma_sum = np.sum(np.log(1 + np.exp(f_i + sqrt2_sigb_xk)))
                    k_exp = np.exp(yf + y_sum_x - log_gamma_sum) * w_ks[k] / np.sqrt(np.pi)
                    k_sum_num = k_sum_num + sqrt2_sigb_xk * k_exp
                    k_sum_den = k_sum_den + k_exp
                b_hat_numerators.append(k_sum_num)
                if k_sum_den == 0.0:
                    b_hat_denominators.append(1)
                else:
                    b_hat_denominators.append(k_sum_den)
            else:
                b_hat_numerators.append(0)
                b_hat_denominators.append(1)
        b_hat = np.array(b_hat_numerators) / np.array(b_hat_denominators)
    elif mode == 'spatial' and y_type == 'continuous':
        gZ_train = get_dummies(X_train['z0'].values, q_spatial)
        D = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
        N = gZ_train.shape[0]
        # increase this as you can
        if X_train.shape[0] > sample_n_train:
            samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        V = gZ_train @ D @ gZ_train.T + np.eye(gZ_train.shape[0]) * sig2e
        V_inv_y = np.linalg.solve(V, y_train.values[samp] - y_pred_tr[samp])
        b_hat = D @ gZ_train.T @ V_inv_y 
    elif mode == 'spatial_embedded':
        loc_df = X_train[['D1', 'D2']]
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_train = last_layer.predict([loc_df])
        if X_train.shape[0] > sample_n_train:
            samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        n_cats = ls
        D_inv = get_D_est(n_cats, 1 / sig2bs_spatial)
        A = gZ_train.T @ gZ_train / sig2e + D_inv
        b_hat = np.linalg.inv(A) @ gZ_train.T / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
        b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
    elif mode == 'survival':
        Hs = weibull_ests[0] * (y_train ** weibull_ests[1])
        b_hat = []
        for i in range(qs[0]):
            i_vec = X_train['z0'] == i
            D_i = X_train['C0'][i_vec].sum()
            A_i = 1 / sig2bs[0] + D_i
            C_i = 1 / sig2bs[0] + np.sum(Hs[i_vec] * np.exp(y_pred_tr[i_vec]))
            b_i = A_i / C_i
            b_hat.append(b_i)
        b_hat = np.array(b_hat)
    return b_hat

def single_random_intercept_b_hat(X_train, y_train, y_pred_tr, qs, sig2e, sig2bs):
    pred_df = pd.DataFrame({'z0': X_train['z0'], 'true': y_train, 'pred': y_pred_tr})
    y_train_bar = pred_df.groupby('z0')['true'].mean()
    y_pred_bar = pred_df.groupby('z0')['pred'].mean()
    ns = pred_df.groupby('z0').size()
    y_train_bar = y_train_bar.reindex(np.arange(qs[0]), fill_value=0)
    y_pred_bar = y_pred_bar.reindex(np.arange(qs[0]), fill_value=0)
    ns = ns.reindex(np.arange(qs[0]), fill_value=0)
    b_hat = ns * sig2bs[0] * (y_train_bar - y_pred_bar) / (sig2e + ns * sig2bs[0])
    b_hat = np.array(b_hat)
    return b_hat

