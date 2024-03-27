import numpy as np
import pandas as pd
from scipy import sparse, stats, special

from tensorflow.keras import Model

from copnn.utils import RegResult, get_cov_mat, get_dummies


def get_D_est(qs, sig2bs):
    D_hat = sparse.eye(np.sum(qs))
    D_hat.setdiag(np.repeat(sig2bs, qs))
    return D_hat

def marginal_inverse(q, marginal):
    if marginal == 'gaussian':
        return stats.norm.ppf(q)
    elif marginal == 'laplace':
        return stats.laplace.ppf(q, scale = 1/np.sqrt(2))
    elif marginal == 'u2':
        return np.sign(q - 0.5) * 2 * np.sqrt(1.5) * (1 - np.sqrt(1 + np.sign(q - 0.5) * (1 - 2*q)))
    elif marginal == 'n2':
        # this is assuming the a parameter is at least 2, otherwise use bisection
        a = 3
        sig = np.sqrt(1 / (1 + a ** 2))
        res = np.zeros(len(q))
        res[q < 0.5] = stats.norm.ppf(2 * q[q < 0.5], loc= -a * sig, scale = sig)
        res[q > 0.5] = stats.norm.ppf(2 * q[q > 0.5] - 1, loc= a * sig, scale = sig)
        res[q == 0.5] = 0
        return res
    elif marginal == 'exponential':
        return -(np.log(1 - q) + 1)
    elif marginal == 'gumbel':
        c = np.sqrt(6) / np.pi
        return stats.gumbel_r.ppf(q, loc = -c * np.euler_gamma, scale = c)
    elif marginal == 'logistic':
        return stats.logistic.ppf(q, scale = np.sqrt(3) / np.pi)
    elif marginal == 'skewnorm':
        alpha = 1
        xi = -alpha * np.sqrt(2 * 1 / (np.pi * (1 + alpha**2) - 2 * alpha**2))
        omega = np.sqrt(np.pi * 1 * (1 + alpha**2) / (np.pi * (1 + alpha**2) - 2 * alpha**2))
        return stats.skewnorm.ppf(q, a = alpha, loc = xi, scale = omega)

def marginal_cdf(x, marginal):
    if marginal == 'gaussian':
        return stats.norm.cdf(x)
    elif marginal == 'laplace':
        return stats.laplace.cdf(x, scale = 1/np.sqrt(2))
    elif marginal == 'u2':
        a = np.sqrt(1.5)
        y = np.clip(x, -2*a + 1e-5, 2*a - 1e-5)
        return 0.5 + y / (2 * a) - np.sign(y) * ((y ** 2)/ (8 * (a ** 2)))
    elif marginal == 'n2':
        a = 3
        sig = 1
        y = x * np.sqrt(1 + a**2) / sig
        y1 = y - a
        y2 = y + a
        return 0.5 * (special.erf(y1 / np.sqrt(2)) + 1)/2 + 0.5 * (special.erf(y2 / np.sqrt(2)) + 1)/2
    elif marginal == 'exponential':
        return 1 - np.exp(-x + x.min())
    elif marginal == 'gumbel':
        c = np.sqrt(6) / np.pi
        return stats.gumbel_r.cdf(x, loc = -c * np.euler_gamma, scale = c)
    elif marginal == 'logistic':
        return stats.logistic.cdf(x, scale = np.sqrt(3) / np.pi)
    elif marginal == 'skewnorm':
        alpha = 1
        xi = -alpha * np.sqrt(2 * 1 / (np.pi * (1 + alpha**2) - 2 * alpha**2))
        omega = np.sqrt(np.pi * 1 * (1 + alpha**2) / (np.pi * (1 + alpha**2) - 2 * alpha**2))
        return stats.skewnorm.cdf(x, a = alpha, loc = xi, scale = omega)

def calc_b_hat(X_train, X_test, y_train, y_pred_tr, qs, q_spatial, sig2e, sig2bs, sig2bs_spatial,
    Z_non_linear, model, ls, mode, rhos, est_cors, dist_matrix, weibull_ests, sample_n_train=10000, copula=False, marginal='gaussian'):
    experimental = False
    if mode in ['intercepts', 'spatial_and_categoricals']:
        if Z_non_linear or len(qs) > 1 or mode == 'spatial_and_categoricals' or copula:
            delta_loc = 0
            if mode == 'spatial_and_categoricals':
                delta_loc = 1
            gZ_trains = []
            gZ_tests = []
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
                if not experimental:
                    # in spatial_and_categoricals increase this as you can
                    if mode == 'spatial_and_categoricals' and X_train.shape[0] > sample_n_train:
                        samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
                    elif X_train.shape[0] > 100000:
                        # Z linear, multiple categoricals, V is relatively sparse, will solve with sparse.linalg.cg
                        # consider sampling or "inducing points" approach if matrix is huge
                        # samp = np.random.choice(X_train.shape[0], 100000, replace=False)
                        pass
                gZ_train = gZ_train.tocsr()[samp]
                gZ_test = gZ_test.tocsr()
            if not experimental:
                D = get_D_est(n_cats, sig2bs)
                V = gZ_train @ D @ gZ_train.T + sparse.eye(gZ_train.shape[0]) * sig2e
                if copula:
                    V /= (np.sum(sig2bs) + sig2e)
                    D /= (np.sum(sig2bs) + sig2e)
                if mode == 'spatial_and_categoricals':
                    gZ_train_spatial = get_dummies(X_train['z0'].values, q_spatial)
                    D_spatial = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
                    gZ_train_spatial = gZ_train_spatial[samp]
                    V += gZ_train_spatial @ D_spatial @ gZ_train_spatial.T
                    gZ_train = sparse.hstack([gZ_train, gZ_train_spatial])
                    D = sparse.block_diag((D, D_spatial))
                    V_inv_y = np.linalg.solve(V, y_train.values[samp] - y_pred_tr[samp])
                else:
                    if copula:
                        if Z_non_linear:
                            V_inv_y = np.linalg.solve(V, (y_train.values[samp] - y_pred_tr[samp])/np.sqrt(np.sum(sig2bs) + sig2e))
                        else:
                            V_inv_y = sparse.linalg.cg(V, (y_train.values[samp] - y_pred_tr[samp])/np.sqrt(np.sum(sig2bs) + sig2e))[0]
                    else:
                        if Z_non_linear:
                            V_inv_y = np.linalg.solve(V, (y_train.values[samp] - y_pred_tr[samp]))
                        else:
                            V_inv_y = sparse.linalg.cg(V, (y_train.values[samp] - y_pred_tr[samp]))[0]
                b_hat = D @ gZ_train.T @ V_inv_y
                if copula:
                    # b_hat = marginal_inverse(stats.norm.cdf(b_hat), marginal) * np.sqrt(np.sum(sig2bs) + sig2e)
                    b_hat_mean = gZ_test @ b_hat
                    # woodbury
                    D_inv = get_D_est(n_cats, (np.sum(sig2bs) + sig2e)/sig2bs)
                    sig2e_rho = sig2e / (np.sum(sig2bs) + sig2e)
                    A = gZ_train.T @ gZ_train / sig2e_rho + D_inv
                    V_inv = sparse.eye(V.shape[0]) / sig2e_rho - (1/(sig2e_rho**2)) * gZ_train @ sparse.linalg.inv(A) @ gZ_train.T
                    b_hat_cov = sparse.eye(gZ_test.shape[0]) - gZ_test @ D @ gZ_train.T @ V_inv @ gZ_train @ D @ gZ_test.T
                    b_hat = []
                    for i in range(gZ_test.shape[0]):
                        b_hat_norm_quantiles = stats.norm.ppf(np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), loc=b_hat_mean[i], scale=b_hat_cov[i,i])
                        b_hat_orig_quantiles = marginal_inverse(stats.norm.cdf(b_hat_norm_quantiles), marginal) * np.sqrt(np.sum(sig2bs) + sig2e)
                        b_hat_i = 0.28871*b_hat_orig_quantiles[3] + 0.18584*(b_hat_orig_quantiles[2] + b_hat_orig_quantiles[4]) + 0.13394*(b_hat_orig_quantiles[1] + b_hat_orig_quantiles[5]) + 0.036128*(b_hat_orig_quantiles[0] + b_hat_orig_quantiles[6])
                        # b_hat_i = -0.3039798 * b_hat_orig_quantiles[2] + 1.3039798 * b_hat_orig_quantiles[3]
                        b_hat.append(b_hat_i)
                    b_hat = np.array(b_hat)
            else:
                if mode == 'spatial_and_categoricals':
                    raise ValueError('experimental inverse not yet implemented in this mode')
                D_inv = get_D_est(n_cats, 1 / sig2bs)
                A = gZ_train.T @ gZ_train / sig2e + D_inv
                b_hat = np.linalg.inv(A.toarray()) @ gZ_train.T / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
                b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
        else:
            b_hat = single_random_intercept_b_hat(X_train, y_train, y_pred_tr, qs, sig2e, sig2bs)
    elif mode == 'longitudinal':
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
        if not experimental:
            D = sparse.kron(cov_mat, sparse.eye(q)) + sig2e * sparse.eye(q * len(sig2bs))
            V = gZ_train @ D @ gZ_train.T + sparse.eye(gZ_train.shape[0]) * sig2e
            if copula:
                V /= (np.sum(sig2bs) + sig2e)
                D /= (np.sum(sig2bs) + sig2e)
            if copula:
                V_inv_y = sparse.linalg.cg(V, (y_train.values - y_pred_tr)/np.sqrt(np.sum(sig2bs) + sig2e))[0]
            else:
                V_inv_y = sparse.linalg.cg(V, y_train.values - y_pred_tr)[0]
            b_hat = D @ gZ_train.T @ V_inv_y
        else:
            D = sparse.kron(cov_mat, np.eye(q)) + sig2e * np.eye(q * len(sig2bs))
            D_inv = np.linalg.inv(D)
            A = gZ_train.T @ gZ_train / sig2e + D_inv
            b_hat = np.linalg.inv(A) @ gZ_train.T / sig2e @ (y_train.values - y_pred_tr)
            b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
        if copula:
            # b_hat = marginal_inverse(stats.norm.cdf(b_hat), marginal) * np.sqrt(np.sum(sig2bs) + sig2e)
            b_hat_mean = gZ_test @ b_hat
            # woodbury
            D_inv = np.linalg.inv(D.toarray()) #get_D_est(n_cats, (np.sum(sig2bs) + sig2e)/sig2bs)
            sig2e_rho = sig2e / (np.sum(sig2bs) + sig2e)
            A = gZ_train.T @ gZ_train / sig2e_rho + D_inv
            V_inv = np.eye(V.shape[0]) / sig2e_rho - (1/(sig2e_rho**2)) * gZ_train @ np.linalg.inv(A) @ gZ_train.T
            b_hat_cov = np.eye(gZ_test.shape[0]) - gZ_test @ D @ gZ_train.T @ V_inv @ gZ_train @ D @ gZ_test.T
            b_hat = []
            for i in range(gZ_test.shape[0]):
                b_hat_norm_quantiles = stats.norm.ppf(np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), loc=b_hat_mean[i], scale=b_hat_cov[i,i])
                b_hat_orig_quantiles = marginal_inverse(stats.norm.cdf(b_hat_norm_quantiles), marginal) * np.sqrt(np.sum(sig2bs) + sig2e)
                b_hat_i = 0.28871*b_hat_orig_quantiles[3] + 0.18584*(b_hat_orig_quantiles[2] + b_hat_orig_quantiles[4]) + 0.13394*(b_hat_orig_quantiles[1] + b_hat_orig_quantiles[5]) + 0.036128*(b_hat_orig_quantiles[0] + b_hat_orig_quantiles[6])
                # b_hat_i = -0.3039798 * b_hat_orig_quantiles[2] + 1.3039798 * b_hat_orig_quantiles[3]
                b_hat.append(b_hat_i)
            b_hat = np.array(b_hat)
    elif mode == 'glmm':
        nGQ = 5
        x_ks, w_ks = np.polynomial.hermite.hermgauss(nGQ)
        a = np.unique(X_train['z0'])
        b_hat_numerators = []
        b_hat_denominators = []
        q = qs[0]
        for i in range(q):
            if i in a:
                i_vec = X_train['z0'] == i
                y_i = y_train.values[i_vec]
                f_i = y_pred_tr[i_vec]
                yf = np.dot(y_i, f_i)
                k_sum_num = 0
                k_sum_den = 0
                for k in range(nGQ):
                    sqrt2_sigb_xk = np.sqrt(2) * np.sqrt(sig2bs[0]) * x_ks[k]
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
    elif mode == 'spatial':
        gZ_train = get_dummies(X_train['z0'].values, q_spatial)
        gZ_test = get_dummies(X_test['z0'].values, q_spatial)
        D = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
        N = gZ_train.shape[0]
        # increase this as you can
        if X_train.shape[0] > sample_n_train:
            samp = np.random.choice(X_train.shape[0], sample_n_train, replace=False)
        else:
            samp = np.arange(X_train.shape[0])
        gZ_train = gZ_train[samp]
        V = gZ_train @ D @ gZ_train.T + np.eye(gZ_train.shape[0]) * sig2e
        if copula:
            V /= (sig2bs_spatial[0] + sig2e)
            D /= (sig2bs_spatial[0] + sig2e)
        V_inv_y = np.linalg.solve(V, y_train.values[samp] - y_pred_tr[samp])
        b_hat = D @ gZ_train.T @ V_inv_y
        # A = gZ_train.T @ gZ_train / sig2e + D_inv
        # A_inv_Zt = np.linalg.inv(A) @ gZ_train.T
        # b_hat = A_inv_Zt / sig2e @ (y_train.values[samp] - y_pred_tr[samp])
        # b_hat = np.asarray(b_hat).reshape(gZ_train.shape[1])
        if copula:
            # b_hat = marginal_inverse(stats.norm.cdf(b_hat), marginal) * np.sqrt(sig2bs_spatial[0] + sig2e)
            b_hat_mean = gZ_test @ b_hat
            # woodbury
            D_inv = np.linalg.inv(D) #get_D_est(n_cats, (np.sum(sig2bs) + sig2e)/sig2bs)
            sig2e_rho = sig2e / (sig2bs_spatial[0] + sig2e)
            A = gZ_train.T @ gZ_train / sig2e_rho + D_inv
            V_inv = np.eye(V.shape[0]) / sig2e_rho - (1/(sig2e_rho**2)) * gZ_train @ np.linalg.inv(A) @ gZ_train.T
            b_hat_cov = np.eye(gZ_test.shape[0]) - gZ_test @ D @ gZ_train.T @ V_inv @ gZ_train @ D @ gZ_test.T
            b_hat = []
            for i in range(gZ_test.shape[0]):
                b_hat_norm_quantiles = stats.norm.ppf(np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), loc=b_hat_mean[i], scale=b_hat_cov[i,i])
                b_hat_orig_quantiles = marginal_inverse(stats.norm.cdf(b_hat_norm_quantiles), marginal) * np.sqrt(np.sum(sig2bs) + sig2e)
                b_hat_i = 0.28871*b_hat_orig_quantiles[3] + 0.18584*(b_hat_orig_quantiles[2] + b_hat_orig_quantiles[4]) + 0.13394*(b_hat_orig_quantiles[1] + b_hat_orig_quantiles[5]) + 0.036128*(b_hat_orig_quantiles[0] + b_hat_orig_quantiles[6])
                # b_hat_i = -0.3039798 * b_hat_orig_quantiles[2] + 1.3039798 * b_hat_orig_quantiles[3]
                b_hat.append(b_hat_i)
            b_hat = np.array(b_hat)
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

