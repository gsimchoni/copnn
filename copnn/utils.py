import pandas as pd
import numpy as np
from collections import namedtuple
from scipy import sparse
from scipy import stats
from scipy.spatial.kdtree import distance_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

SimResult = namedtuple('SimResult',
                       ['N', 'test_size', 'pred_unknown', 'batch', 'sig2e', 'sig2bs', 'qs', 'deep', 'iter_id',
                        'exp_type', 'mse', 'sig2e_est', 'sig2b_ests', 'n_epochs', 'time'])

RegResult = namedtuple('RegResult', ['metric_mse', 'metric_mae', 'metric_trim', 'metric_r2', 'sigmas', 'rho_est', 'rhos', 'nll_tr',
                                     'nll_te', 'n_epochs', 'time'])

RegData = namedtuple('RegData', ['X_train', 'X_test', 'y_train', 'y_test',
                                 'x_cols', 'dist_matrix', 'time2measure_dict', 'b_true'])

RegInput = namedtuple('RegInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                   'dist_matrix', 'time2measure_dict', 'b_true',
                                   'N', 'test_size', 'pred_unknown', 'qs', 'sig2e', 'sig2bs', 'rhos', 'sig2bs_spatial',
                                   'q_spatial', 'k', 'batch', 'epochs', 'patience',
                                   'Z_non_linear', 'Z_embed_dim_pct', 'mode',
                                   'n_sig2bs', 'n_sig2bs_spatial', 'estimated_cors',
                                   'verbose', 'n_neurons', 'dropout', 'activation',
                                   'spatial_embed_neurons', 'log_params',
                                   'weibull_lambda', 'weibull_nu', 'resolution', 'shuffle',
                                   'true_marginal', 'fit_marginal'])

def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(vec_size, vec_max), dtype=np.uint16)
    return Z

def get_dummies_np(vec, vec_max):
    vec_size = vec.size
    Z = np.zeros((vec_size, vec_max), dtype=np.uint8)
    Z[np.arange(vec_size), vec] = 1
    return Z

def get_cov_mat(sig2bs, rhos, est_cors):
    cov_mat = np.zeros((len(sig2bs), len(sig2bs)))
    for k in range(len(sig2bs)):
        for j in range(len(sig2bs)):
            if k == j:
                cov_mat[k, j] = sig2bs[k]
            else:
                rho_symbol = ''.join(map(str, sorted([k, j])))
                if rho_symbol in est_cors:
                    rho = rhos[est_cors.index(rho_symbol)]
                else:
                    rho = 0
                cov_mat[k, j] = rho * np.sqrt(sig2bs[k]) * np.sqrt(sig2bs[j])
    return cov_mat

def random_n2(n, sig2, n_sig):
  classes = np.random.binomial(n = 1, p = 0.5, size=n)
  n1 = classes.sum()
  n2 = n - n1
  z = np.zeros(n)
  if n1 > 0:
    z[classes == 1] = np.random.normal(loc = -n_sig * np.sqrt(sig2), scale = np.sqrt(sig2), size = n1)
  if n2 > 0:
    z[classes == 0] = np.random.normal(loc = n_sig * np.sqrt(sig2), scale = np.sqrt(sig2), size = n2)
  return z

def copulize(P, sig2, marginal):
    U = stats.norm.cdf(P)
    if marginal == 'gaussian':
        b = stats.norm.ppf(U)
    elif marginal == 'laplace':
        b = stats.laplace.ppf(U, scale = 1/np.sqrt(2))
    elif marginal == 'u2':
        b = np.sign(U - 0.5) * 2 * np.sqrt(1.5) * (1 - np.sqrt(1 + np.sign(U - 0.5) * (1 - 2*U)))
    elif marginal == 'n2':
        # this is assuming the a parameter is at least 2, otherwise use bisection
        a = 3
        sig = np.sqrt(1 / (1 + a ** 2))
        b = np.zeros(len(U))
        b[U < 0.5] = stats.norm.ppf(2 * U[U < 0.5], loc= -a * sig, scale = sig)
        b[U > 0.5] = stats.norm.ppf(2 * U[U > 0.5] - 1, loc= a * sig, scale = sig)
    elif marginal == 'exponential':
        b = -(np.log(1 - U) + 1)
    elif marginal == 'gumbel':
        c = np.sqrt(6) / np.pi
        b = stats.gumbel_r.ppf(U, loc = -c * np.euler_gamma, scale = c)
    elif marginal == 'logistic':
        b = stats.logistic.ppf(U, scale = np.sqrt(3) / np.pi)
    elif marginal == 'skewnorm':
        alpha = 1
        xi = -alpha * np.sqrt(2 * 1 / (np.pi * (1 + alpha**2) - 2 * alpha**2))
        omega = np.sqrt(np.pi * 1 * (1 + alpha**2) / (np.pi * (1 + alpha**2) - 2 * alpha**2))
        b = stats.skewnorm.ppf(U, a = alpha, loc = xi, scale = omega)
    return b * np.sqrt(sig2)

def generate_data(mode, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos, marginal, test_size, pred_unknown_clusters, params):
    if marginal not in ['gaussian', 'laplace', 'exponential', 'u2', 'n2', 'gumbel', 'logistic', 'skewnorm']:
        raise ValueError(marginal + ' is unknown marginal distribution')
    n_fixed_effects = params['n_fixed_effects']
    X = np.random.uniform(-1, 1, N * n_fixed_effects).reshape((N, n_fixed_effects))
    betas = np.ones(n_fixed_effects)
    Xbeta = params['fixed_intercept'] + X @ betas
    dist_matrix = None
    time2measure_dict = None
    if params['X_non_linear']:
        fX = Xbeta * np.cos(Xbeta) + 2 * X[:, 0] * X[:, 1]
    else:
        fX = Xbeta
    df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(n_fixed_effects)]
    df.columns = x_cols
    y = fX
    e = np.random.normal(0, np.sqrt(sig2e), N)
    if mode in ['intercepts', 'glmm', 'spatial_and_categoricals']:
        sum_gZbs = 0
        delta_loc = 0
        if mode == 'spatial_and_categoricals':
            delta_loc = 1
        for k, q in enumerate(qs):
            fs = np.random.poisson(params['n_per_cat'], q) + 1
            fs_sum = fs.sum()
            ps = fs/fs_sum
            ns = np.random.multinomial(N, ps)
            Z_idx = np.repeat(range(q), ns)
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
            df['z' + str(k + delta_loc)] = Z_idx
        b_copula = copulize((sum_gZbs + e)/np.sqrt(sig2e + np.sum(sig2bs)), sig2e + np.sum(sig2bs), marginal)
        y = y + b_copula    
    if mode == 'slopes': # len(qs) should be 1
        fs = np.random.poisson(params['n_per_cat'], qs[0]) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(N, ps)
        Z_idx = np.repeat(range(qs[0]), ns)
        max_period = np.arange(ns.max())
        t = np.concatenate([max_period[:k] for k in ns]) / max_period[-1]
        estimated_cors = [] if params['estimated_cors'] is None else params['estimated_cors']
        cov_mat = get_cov_mat(sig2bs, rhos, estimated_cors)
        bs = np.random.multivariate_normal(np.zeros(len(sig2bs)), cov_mat, qs[0])
        b = bs.reshape((qs[0] * len(sig2bs),), order = 'F')
        Z0 = sparse.csr_matrix(get_dummies(Z_idx, qs[0]))
        Z_list = [Z0]
        for k in range(1, len(sig2bs)):
            y += t ** k # fixed part t + t^2 + t^3 + ...
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Zb = sparse.hstack(Z_list) @ b
        y = y + Zb
        df['t'] = t
        df['z0'] = Z_idx
        x_cols.append('t')
        time2measure_dict = {t: i for i, t in enumerate(np.sort(df['t'].unique()))}
    if mode in ['spatial', 'spatial_embedded', 'spatial_and_categoricals']:
        coords = np.stack([np.random.uniform(-10, 10, q_spatial), np.random.uniform(-10, 10, q_spatial)], axis=1)
        # ind = np.lexsort((coords[:, 1], coords[:, 0]))    
        # coords = coords[ind]
        dist_matrix = squareform(pdist(coords)) ** 2
        D = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
        b = np.random.multivariate_normal(np.zeros(q_spatial), D, 1)[0]
        fs = np.random.poisson(params['n_per_cat'], q_spatial) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(N, ps)
        Z_idx = np.repeat(range(q_spatial), ns)
        gZb = np.repeat(b, ns)
        df['z0'] = Z_idx
        b_copula = copulize((gZb + e)/np.sqrt(sig2e + sig2bs_spatial[0]), sig2e + sig2bs_spatial[0], marginal)
        y = y + b_copula
        coords_df = pd.DataFrame(coords[Z_idx])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
        df = pd.concat([df, coords_df], axis=1)
        x_cols.extend(co_cols)
    if mode == 'glmm':
        p = np.exp(y)/(1 + np.exp(y))
        y = np.random.binomial(1, p, size=N)
    df['y'] = y
    pred_future = params['longitudinal_predict_future'] if 'longitudinal_predict_future' in params and mode == 'slopes' else False
    if  pred_future:
        # test set is "the future" or those obs with largest t
        df.sort_values('t', inplace=True)
    if pred_unknown_clusters:
        if mode in ['spatial', 'spatial_fit_categorical', 'spatial_and_categorical']:
            cluster_q = q_spatial
        else:
            cluster_q = qs[0]
        train_clusters, test_clusters = train_test_split(range(cluster_q), test_size=test_size)
        X_train = df[df['z0'].isin(train_clusters)]
        X_test = df[df['z0'].isin(test_clusters)]
        y_train = df['y'][df['z0'].isin(train_clusters)]
        y_test = df['y'][df['z0'].isin(test_clusters)]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('y', axis=1), df['y'], test_size=test_size, shuffle=not pred_future)
    return RegData(X_train, X_test, y_train, y_test, x_cols, dist_matrix, time2measure_dict, b_copula)
