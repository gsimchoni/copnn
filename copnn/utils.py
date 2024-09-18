import pandas as pd
import numpy as np
from collections import namedtuple
from scipy import sparse, stats

SimResult = namedtuple('SimResult',
                       ['N', 'test_size', 'pred_unknown', 'batch', 'sig2e', 'sig2bs', 'qs', 'deep', 'iter_id',
                        'exp_type', 'mse', 'sig2e_est', 'sig2b_ests', 'n_epochs', 'time'])

RegResult = namedtuple('RegResult', ['metric_mse_no_re', 'metric_mse', 'metric_mae',
                                     'metric_mse_trim', 'metric_r2',
                                     'sigmas', 'sig_ratio', 'rhos', 'nll_tr',
                                     'nll_te', 'n_epochs', 'time'])

RegData = namedtuple('RegData', ['X_train', 'X_test', 'y_train', 'y_test',
                                 'x_cols', 'dist_matrix', 'time2measure_dict', 'b_true'])

RegInput = namedtuple('RegInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                   'dist_matrix', 'time2measure_dict', 'b_true',
                                   'N', 'test_size', 'pred_unknown', 'qs', 'sig2e', 'sig2bs', 'rhos', 'sig2bs_spatial',
                                   'q_spatial', 'k', 'batch', 'epochs', 'patience',
                                   'Z_non_linear', 'Z_embed_dim_pct', 'mode', 'y_type',
                                   'n_sig2bs', 'n_sig2bs_spatial', 'estimated_cors',
                                   'verbose', 'n_neurons', 'dropout', 'activation',
                                   'spatial_embed_neurons', 'log_params',
                                   'weibull_lambda', 'weibull_nu', 'resolution', 'shuffle',
                                   'true_dist', 'fit_dist'])

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

def sample_ns(N, q, n_per_cat):
        fs = np.random.poisson(n_per_cat, q) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(N, ps)
        return ns

def copulize(z, distribution, sig2):
    u = stats.norm.cdf(z)
    b = distribution.quantile(u)
    return b * np.sqrt(sig2)
