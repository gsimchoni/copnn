import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

from copnn.utils import get_cov_mat, get_dummies, sample_ns, copulize, RegData

class Mode:
    def __init__(self, mode_par):
        self.mode_par = mode_par
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.mode_par == other
        else:
            return NotImplemented
    
    def __str__(self):
        return self.mode_par
    
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

    def create_df(self, X, y, n_fixed_effects, Z_idx_list, t, coords):
        time2measure_dict = None
        df = pd.DataFrame(X)
        x_cols = ['X' + str(i) for i in range(n_fixed_effects)]
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
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')


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
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')


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
    
    def create_df(self, X, y, n_fixed_effects, Z_idx_list, t, coords):
        df, x_cols, time2measure_dict = super().create_df(X, y, n_fixed_effects, Z_idx_list, t, coords)
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
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')


class Spatial(Mode):
    def __init__(self):
        super().__init__('spatial')
    
    def sample_re(self, params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos):
        _, _, _, t, time_fe, _, _ = super().sample_re()
        coords = np.stack([np.random.uniform(-10, 10, q_spatial), np.random.uniform(-10, 10, q_spatial)], axis=1)
        # ind = np.lexsort((coords[:, 1], coords[:, 0]))    
        # coords = coords[ind]
        dist_matrix = squareform(pdist(coords)) ** 2
        D = sig2bs_spatial[0] * np.exp(-dist_matrix / (2 * sig2bs_spatial[1]))
        b = np.random.multivariate_normal(np.zeros(q_spatial), D, 1)[0]
        ns = sample_ns(N, q_spatial, params['n_per_cat'])
        Z_idx = np.repeat(range(q_spatial), ns)
        gZb = np.repeat(b, ns)
        sig2 = sig2e + sig2bs_spatial[0]
        return gZb, time_fe, sig2, [Z_idx], t, coords, dist_matrix
    
    def create_df(self, X, y, n_fixed_effects, Z_idx_list, t, coords):
        df, x_cols, time2measure_dict = super().create_df(X, y, n_fixed_effects, Z_idx_list, t, coords)
        coords_df = pd.DataFrame(coords[Z_idx_list[0]])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
        df = pd.concat([df, coords_df], axis=1)
        return df, x_cols, time2measure_dict
    
    def train_test_split(self, df, test_size, pred_unknown_clusters, params, qs, q_spatial):
        return super().train_test_split(df, test_size, pred_unknown_clusters, q_spatial, False)
    
    def V_batch(self):
        raise NotImplementedError('The V_batch method is not implemented.')
    
    def m_batch(self):
        raise NotImplementedError('The m_batch method is not implemented.')
    
    def predict_re(self):
        raise NotImplementedError('The predict_re method is not implemented.')
    
    def predict_y(self):
        raise NotImplementedError('The predict_y method is not implemented.')
    
    def build_net_input(self):
        raise NotImplementedError('The build_net_input method is not implemented.')

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

def generate_data(mode, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos,
                  distribution, test_size, pred_unknown_clusters, params):
    X, fX = mode.sample_fe(params, N)
    e = np.random.normal(0, np.sqrt(sig2e), N)
    Zb, time_fe, sig2, Z_idx_list, t, coords, dist_matrix = mode.sample_re(params, qs, sig2e, sig2bs, sig2bs_spatial, q_spatial, N, rhos)
    z = (Zb + e)/np.sqrt(sig2)
    b_cop = copulize(z, distribution, sig2)
    y = fX + time_fe + b_cop
    df, x_cols, time2measure_dict = mode.create_df(X, y, params['n_fixed_effects'], Z_idx_list, t, coords)
    X_train, X_test, y_train, y_test = mode.train_test_split(df, test_size, pred_unknown_clusters, params, qs, q_spatial)
    return RegData(X_train, X_test, y_train, y_test, x_cols, dist_matrix, time2measure_dict, b_cop)
