import logging
import os
from itertools import product

import pandas as pd

from copnn.distributions import get_distribution
from copnn.modes.categorical import Categorical
from copnn.modes.longitudinal import Longitudinal
from copnn.modes.mode import generate_data
from copnn.modes.spatial import Spatial
from copnn.regression import run_regression
from copnn.utils import RegInput

logger = logging.getLogger('COPNN.logger')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment to disable gpu

class Count:
    curr = 0

    def __init__(self, startWith=None):
        if startWith is not None:
            Count.curr = startWith - 1

    def gen(self):
        while True:
            Count.curr += 1
            yield Count.curr


def iterate_reg_types(counter, res_df, out_file, reg_in, reg_types, verbose):
    for reg_type in reg_types:
        if verbose:
            logger.info(f'mode {reg_type}')
        res = run_reg(reg_in, reg_type)
        res_summary = summarize_sim(reg_in, res, reg_type)
        res_df.loc[next(counter)] = res_summary
        logger.debug(f'  Finished {reg_type}.')
    res_df.to_csv(out_file, float_format='%.6g')

def run_reg(reg_in, reg_type):
    return run_regression(reg_in.X_train, reg_in.X_test, reg_in.y_train,
        reg_in.y_test, reg_in.qs, reg_in.q_spatial,
        reg_in.x_cols, reg_in.batch, reg_in.epochs, reg_in.patience,
        reg_in.n_neurons, reg_in.dropout, reg_in.activation, reg_type=reg_type,
        Z_non_linear=reg_in.Z_non_linear, Z_embed_dim_pct = reg_in.Z_embed_dim_pct,
        mode = reg_in.mode, y_type = reg_in.y_type, n_sig2bs = reg_in.n_sig2bs,
        n_sig2bs_spatial = reg_in.n_sig2bs_spatial, est_cors = reg_in.estimated_cors,
        dist_matrix = reg_in.dist_matrix, time2measure_dict = reg_in.time2measure_dict,
        spatial_embed_neurons = reg_in.spatial_embed_neurons, resolution=reg_in.resolution,
        verbose = reg_in.verbose, log_params = reg_in.log_params, idx = reg_in.k,
        shuffle = reg_in.shuffle, fit_dist = reg_in.fit_dist, b_true = reg_in.b_true)


def summarize_sim(reg_in, res, reg_type):
    if reg_in.q_spatial is not None:
        q_spatial = [reg_in.q_spatial]
    else:
        q_spatial = []
    res = [reg_in.mode, reg_in.N, reg_in.test_size, reg_in.batch, reg_in.pred_unknown, reg_in.sig2e] +\
        list(reg_in.sig2bs) + list(reg_in.sig2bs_spatial) +\
        list(reg_in.qs) + list(reg_in.rhos) + q_spatial + [str(reg_in.true_dist), str(reg_in.fit_dist)] +\
        [reg_in.k, reg_type, res.metric_mse_no_re, res.metric_mse, res.metric_mse_blup,
         res.metric_mae, res.metric_mae_blup, res.metric_mse_trim, res.metric_mse_trim_blup,
         res.metric_r2, res.metric_r2_blup, res.sigmas[0]] +\
        res.sigmas[1] + res.sigmas[2] + res.rhos + [res.sig_ratio] +\
        [res.nll_tr, res.nll_te] + [res.n_epochs, res.time]
    return res


def get_mode(mode_par):
    if mode_par == 'categorical':
        mode = Categorical()
    elif mode_par == 'longitudinal':
        mode = Longitudinal()
    elif mode_par == 'spatial':
        mode = Spatial()
    else:
        raise NotImplementedError(f'{mode_par} mode not implemented.')
    return mode


def simulation(out_file, params):
    mode = get_mode(params['mode'])
    y_type = params['y_type']
    counter = Count().gen()
    n_sig2bs = len(params['sig2b_list'])
    n_sig2bs_spatial = len(params['sig2b_spatial_list'])
    n_categoricals = len(params['q_list'])
    n_rhos = len([] if params['rho_list'] is None else params['rho_list'])
    estimated_cors = [] if params['estimated_cors'] is None else params['estimated_cors']
    spatial_embed_out_dim_name = []
    rhos_names =  []
    rhos_est_names =  []
    sig2bs_spatial_names = []
    sig2bs_spatial_est_names = []
    q_spatial_name = []
    q_spatial_list = [None]
    metric = 'mse'
    resolution = None
    shuffle = params['shuffle'] if 'shuffle' in params else False
    if mode == 'categorical':
        assert n_sig2bs == n_categoricals
    elif mode == 'longitudinal':
        assert n_categoricals == 1
        # assert n_rhos == len(estimated_cors)
        rhos_names =  list(map(lambda x: 'rho' + str(x), range(n_rhos)))
        rhos_est_names =  list(map(lambda x: 'rho_est' + str(x), range(len(estimated_cors))))
    elif mode == 'glmm':
        assert n_categoricals == 1
        assert n_sig2bs == n_categoricals
        metric = 'auc'
    elif mode == 'spatial':
        assert n_categoricals == 0
        assert n_sig2bs == 0
        assert n_sig2bs_spatial == 2
        sig2bs_spatial_names = ['sig2b0_spatial', 'sig2b1_spatial']
        sig2bs_spatial_est_names = ['sig2b_spatial_est0', 'sig2b_spatial_est1']
        q_spatial_name = ['q_spatial']
        q_spatial_list = params['q_spatial_list']
        if 'resolution' in params:
            resolution = params['resolution']
    elif mode == 'spatial_and_categoricals':
        assert n_sig2bs == n_categoricals
        assert n_sig2bs_spatial == 2
        sig2bs_spatial_names = ['sig2b0_spatial', 'sig2b1_spatial']
        sig2bs_spatial_est_names = ['sig2b_spatial_est0', 'sig2b_spatial_est1']
        q_spatial_name = ['q_spatial']
        q_spatial_list = params['q_spatial_list']
        if 'resolution' in params:
            resolution = params['resolution']
    elif mode == 'spatial_embedded':
        assert n_categoricals == 0
        assert n_sig2bs == 0
        assert n_sig2bs_spatial == 2
        sig2bs_spatial_names = ['sig2b0_spatial', 'sig2b1_spatial']
        sig2bs_spatial_est_names = ['sig2b_spatial_est0', 'sig2b_spatial_est1']
        spatial_embed_out_dim_name = ['spatial_embed_out_dim']
        q_spatial_name = ['q_spatial']
        q_spatial_list = params['q_spatial_list']
    else:
        raise ValueError('Unknown mode')
    qs_names =  list(map(lambda x: 'q' + str(x), range(n_categoricals)))
    sig2bs_names =  list(map(lambda x: 'sig2b' + str(x), range(n_sig2bs)))
    sig2bs_est_names =  list(map(lambda x: 'sig2b_est' + str(x), range(n_sig2bs)))
    test_size = params.get('test_size', 0.2)
    pred_unknown_clusters = params.get('pred_unknown_clusters', False)
    
    res_df = pd.DataFrame(columns=['mode', 'N', 'test_size', 'batch', 'pred_unknown', 'sig2e'] +\
                          sig2bs_names + sig2bs_spatial_names + qs_names + rhos_names + q_spatial_name + ['true_marginal', 'fit_marginal'] +\
                            ['experiment', 'exp_type', 'mse_no_re', metric, 'mse_blup', 'mae', 'mae_blup',
                             'mse_trim', 'mse_trim_blup', 'r2', 'r2_blup', 'sig2e_est'] +\
                                sig2bs_est_names + rhos_est_names + sig2bs_spatial_est_names +\
                                    ['sig_ratio', 'nll_train', 'nll_test'] + ['n_epochs', 'time'])
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for qs in product(*params['q_list']):
                for sig2bs in product(*params['sig2b_list']):
                    for rhos in product(*params['rho_list']):
                        for sig2bs_spatial in product(*params['sig2b_spatial_list']):
                            for q_spatial in q_spatial_list:
                                for true_marginal in params['true_marginal']:
                                    for fit_marginal in params['fit_marginal']:
                                        logger.info(f'mode: {mode}, distribution: {true_marginal}, N: {N}, test: {test_size:.2f}, qs: {", ".join(map(str, qs))}, '
                                                                f'sig2e: {sig2e}, '
                                                                f'sig2bs_mean: {", ".join(map(str, sig2bs))}')
                                        for k in range(params['n_iter']):
                                            true_dist = get_distribution(true_marginal)
                                            fit_dist = get_distribution(fit_marginal)
                                            reg_data = generate_data(
                                                mode, y_type, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial,
                                                N, rhos, true_dist, test_size, pred_unknown_clusters, params)
                                            logger.info(f' iteration: {k}')
                                            reg_in = RegInput(*reg_data, N, test_size, pred_unknown_clusters, qs, sig2e,
                                                            sig2bs, rhos, sig2bs_spatial, q_spatial, k, params['batch'], params['epochs'], params['patience'],
                                                            params['Z_non_linear'], params['Z_embed_dim_pct'], mode,
                                                            y_type, n_sig2bs, n_sig2bs_spatial,
                                                            estimated_cors, params['verbose'],
                                                            params['n_neurons'], params['dropout'], params['activation'],
                                                            params['spatial_embed_neurons'], params['log_params'],
                                                            params['weibull_lambda'], params['weibull_nu'], resolution, shuffle, true_dist, fit_dist)
                                            iterate_reg_types(counter, res_df, out_file, reg_in, params['exp_types'], params['verbose'])