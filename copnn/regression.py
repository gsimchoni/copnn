import time
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Concatenate,\
    Reshape, Input, Masking, LSTM, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras import Model
import torch

from copnn.utils import RegResult, get_dummies
from copnn.callbacks import LogEstParams
from copnn.lmmnll import LMMNLL
from copnn.copnll import COPNLL
from copnn.calc_b_hat import *


def add_layers_sequential(model, n_neurons, dropout, activation, input_dim):
    if len(n_neurons) > 0:
        model.add(Dense(n_neurons[0], input_dim=input_dim, activation=activation))
        if dropout is not None and len(dropout) > 0:
            model.add(Dropout(dropout[0]))
        for i in range(1, len(n_neurons) - 1):
            model.add(Dense(n_neurons[i], activation=activation))
            if dropout is not None and len(dropout) > i:
                model.add(Dropout(dropout[i]))
        if len(n_neurons) > 1:
            model.add(Dense(n_neurons[-1], activation=activation))


def add_layers_functional(X_input, n_neurons, dropout, activation, input_dim):
    if len(n_neurons) > 0:
        x = Dense(n_neurons[0], input_dim=input_dim, activation=activation)(X_input)
        if dropout is not None and len(dropout) > 0:
            x = Dropout(dropout[0])(x)
        for i in range(1, len(n_neurons) - 1):
            x = Dense(n_neurons[i], activation=activation)(x)
            if dropout is not None and len(dropout) > i:
                x = Dropout(dropout[i])(x)
        if len(n_neurons) > 1:
            x = Dense(n_neurons[-1], activation=activation)(x)
        return x
    return X_input


def process_one_hot_encoding(X_train, X_test, x_cols):
    z_cols = X_train.columns[X_train.columns.str.startswith('z')]
    X_train_new = X_train[x_cols]
    X_test_new = X_test[x_cols]
    for z_col in z_cols:
        X_train_ohe = pd.get_dummies(X_train[z_col])
        X_test_ohe = pd.get_dummies(X_test[z_col])
        X_test_cols_in_train = set(X_test_ohe.columns).intersection(X_train_ohe.columns)
        X_train_cols_not_in_test = set(X_train_ohe.columns).difference(X_test_ohe.columns)
        X_test_comp = pd.DataFrame(np.zeros((X_test.shape[0], len(X_train_cols_not_in_test))),
            columns=list(X_train_cols_not_in_test), dtype=np.uint8, index=X_test.index)
        X_test_ohe_comp = pd.concat([X_test_ohe[list(X_test_cols_in_train)], X_test_comp], axis=1)
        X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
        X_train_ohe.columns = list(map(lambda c: z_col + '_' + str(c), X_train_ohe.columns))
        X_test_ohe_comp.columns = list(map(lambda c: z_col + '_' + str(c), X_test_ohe_comp.columns))
        X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
        X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
    return X_train_new, X_test_new

def run_reg_ohe_or_ignore(X_train, X_test, y_train, y_test, qs, x_cols, batch_size, epochs,
        patience, n_neurons, dropout, activation, mode,
        n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False, ignore_RE=False):
    if mode == 'glmm':
        loss = 'binary_crossentropy'
        last_layer_activation = 'sigmoid'
    else:
        loss = 'mse'
        last_layer_activation = 'linear'
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(X_train, X_test, x_cols)

    model = Sequential()
    add_layers_sequential(model, n_neurons, dropout, activation, X_train.shape[1])
    model.add(Dense(1, activation=last_layer_activation))

    model.compile(loss=loss, optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, callbacks=callbacks, verbose=verbose)
    y_pred = model.predict(X_test, verbose=verbose).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    return y_pred, (None, none_sigmas, none_sigmas_spatial), none_rhos, len(history.history['loss']), None, None

def run_lmmnn(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
        mode, n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons,
        verbose=False, Z_non_linear=False, Z_embed_dim_pct=10, log_params=False, idx=0, shuffle=False, sample_n_train=10000, b_true=None):
    if mode in ['spatial', 'spatial_embedded', 'spatial_and_categoricals']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
    if mode == 'survival':
        x_cols = [x_col for x_col in x_cols if x_col not in ['C0']]
    # dmatrix_tf = tf.constant(dist_matrix)
    dmatrix_tf = dist_matrix
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        if mode in ['spatial']:
            n_sig2bs_init = n_sig2bs_spatial
            n_RE_inputs = 1
        elif mode == 'spatial_and_categoricals':
            n_sig2bs_init = n_sig2bs_spatial + len(qs)
            n_RE_inputs = 1 + len(qs)
        else:
            n_sig2bs_init = len(qs)
            n_RE_inputs = len(qs)
        for _ in range(n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
    elif mode == 'slopes':
        z_cols = ['z0', 't']
        n_RE_inputs = 2
        n_sig2bs_init = n_sig2bs
        Z_input = Input(shape=(1,), dtype=tf.int64)
        t_input = Input(shape=(1,))
        Z_inputs = [Z_input, t_input]
    elif mode == 'spatial_embedded':
        Z_inputs = [Input(shape=(2,))]
        n_sig2bs_init = 1
    elif mode == 'survival':
        z_cols = ['z0', 'C0']
        Z_input = Input(shape=(1,), dtype=tf.int64)
        event_input = Input(shape=(1,))
        Z_inputs = [Z_input, event_input]
        n_sig2bs_init = 1
    
    out_hidden = add_layers_functional(X_input, n_neurons, dropout, activation, X_train[x_cols].shape[1])
    y_pred_output = Dense(1)(out_hidden)
    if Z_non_linear and (mode in ['intercepts', 'glmm', 'survival']):
        Z_nll_inputs = []
        ls = []
        for k, q in enumerate(qs):
            l = int(q * Z_embed_dim_pct / 100.0)
            Z_embed = Embedding(q, l, input_length=1, name='Z_embed' + str(k))(Z_inputs[k])
            Z_embed = Reshape(target_shape=(l, ))(Z_embed)
            Z_nll_inputs.append(Z_embed)
            ls.append(l)
    elif mode == 'spatial_embedded':
        Z_embed = add_layers_functional(Z_inputs[0], spatial_embed_neurons, dropout=None, activation='relu', input_dim=2)
        Z_nll_inputs = [Z_embed]
        ls = [spatial_embed_neurons[-1]]
        Z_non_linear = True
    else:
        Z_nll_inputs = Z_inputs
        ls = None
    sig2bs_init = np.ones(n_sig2bs_init, dtype=np.float32)
    rhos_init = np.zeros(len(est_cors), dtype=np.float32)
    weibull_init = np.ones(2, dtype=np.float32)
    nll = LMMNLL(mode, 1.0, sig2bs_init, rhos_init, weibull_init, est_cors, Z_non_linear, dmatrix_tf)(
        y_true_input, y_pred_output, Z_nll_inputs)
    model = Model(inputs=[X_input, y_true_input] + Z_inputs, outputs=nll)

    model.compile(optimizer='adam')

    patience = epochs if patience is None else patience
    if Z_non_linear and mode == 'intercepts':
        # in complex scenarios such as non-linear g(Z) consider training "more", until var components norm has converged
        # callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    else:
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    if log_params:
        callbacks.extend([LogEstParams(idx), CSVLogger('res_params.csv', append=True)])
    if not Z_non_linear:
        X_train.sort_values(by=z_cols, inplace=True)
        y_train = y_train[X_train.index]
    if mode == 'spatial_embedded':
        X_train_z_cols = [X_train[['D1', 'D2']]]
        X_test_z_cols = [X_test[['D1', 'D2']]]
    else:
        X_train_z_cols = [X_train[z_col] for z_col in z_cols]
        X_test_z_cols = [X_test[z_col] for z_col in z_cols]
    history = model.fit([X_train[x_cols], y_train] + X_train_z_cols, None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose, shuffle=shuffle)
    nll_tr = model.evaluate([X_train[x_cols], y_train] + X_train_z_cols, verbose=verbose)
    nll_te = model.evaluate([X_test[x_cols], y_test] + X_test_z_cols, verbose=verbose)

    sig2e_est, sig2b_ests, rho_ests, weibull_ests = model.layers[-1].get_vars()
    if mode in ['spatial', 'spatial_embedded']:
        sig2b_spatial_ests = sig2b_ests
        sig2b_ests = []
    elif mode == 'spatial_and_categoricals':
        sig2b_spatial_ests = sig2b_ests[:2]
        sig2b_ests = sig2b_ests[2:]
    else:
        sig2b_spatial_ests = []
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols, verbose=verbose).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, q_spatial, sig2e_est, sig2b_ests, sig2b_spatial_ests,
                Z_non_linear, model, ls, mode, rho_ests, est_cors, dist_matrix, weibull_ests, sample_n_train)
    dummy_y_test = np.random.normal(size=y_test.shape)
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        if Z_non_linear or len(qs) > 1 or mode == 'spatial_and_categoricals':
            delta_loc = 0
            if mode == 'spatial_and_categoricals':
                delta_loc = 1
            Z_tests = []
            for k, q in enumerate(qs):
                Z_test = get_dummies(X_test['z' + str(k + delta_loc)], q)
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    Z_test = Z_test @ W_est
                Z_tests.append(Z_test)
            if Z_non_linear:
                Z_test = np.hstack(Z_tests)
            else:
                Z_test = sparse.hstack(Z_tests)
            if mode == 'spatial_and_categoricals':
                Z_test = sparse.hstack([Z_test, get_dummies(X_test['z0'], q_spatial)])
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
        else:
            # if model input is that large, this 2nd call to predict may cause OOM due to GPU memory issues
            # if that is the case use tf.convert_to_tensor() explicitly with a call to model() without using predict() method
            # y_pred = model([tf.convert_to_tensor(X_test[x_cols]), tf.convert_to_tensor(dummy_y_test), tf.convert_to_tensor(X_test_z_cols[0])], training=False).numpy().reshape(
            #     X_test.shape[0]) + b_hat[X_test['z0']]
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) #+ b_hat[X_test['z0']]
        if mode == 'glmm':
            y_pred = np.exp(y_pred)/(1 + np.exp(y_pred))
    elif mode == 'slopes':
        q = qs[0]
        Z0 = get_dummies(X_test['z0'], q)
        t = X_test['t'].values
        N = X_test.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2b_ests)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Z_test = sparse.hstack(Z_list)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
    elif mode == 'spatial_embedded':
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_test = last_layer.predict(X_test_z_cols, verbose=verbose)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + gZ_test @ b_hat
        sig2b_spatial_ests = np.concatenate([sig2b_spatial_ests, [np.nan]])
    elif mode == 'survival':
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
        y_pred = y_pred + np.log(b_hat[X_test['z0']])
    return y_pred, (sig2e_est, list(sig2b_ests), list(sig2b_spatial_ests)), list(rho_ests), len(history.history['loss']), nll_tr, nll_te


def run_copnn(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
        mode, n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons, fit_marginal,
        verbose=False, Z_non_linear=False, Z_embed_dim_pct=10, log_params=False, idx=0, shuffle=False, sample_n_train=10000, b_true=None):
    if mode in ['spatial', 'spatial_embedded', 'spatial_and_categoricals']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2']]
    # dmatrix_tf = tf.constant(dist_matrix)
    dmatrix_tf = dist_matrix
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        if mode == 'spatial':
            n_sig2bs_init = n_sig2bs_spatial
            n_RE_inputs = 1
        elif mode == 'spatial_and_categoricals':
            n_sig2bs_init = n_sig2bs_spatial + len(qs)
            n_RE_inputs = 1 + len(qs)
        else:
            n_sig2bs_init = len(qs)
            n_RE_inputs = len(qs)
        for _ in range(n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
    elif mode == 'slopes':
        z_cols = ['z0', 't']
        n_RE_inputs = 2
        n_sig2bs_init = n_sig2bs
        Z_input = Input(shape=(1,), dtype=tf.int64)
        t_input = Input(shape=(1,))
        Z_inputs = [Z_input, t_input]
    elif mode == 'spatial_embedded':
        Z_inputs = [Input(shape=(2,))]
        n_sig2bs_init = 1
    
    out_hidden = add_layers_functional(X_input, n_neurons, dropout, activation, X_train[x_cols].shape[1])
    y_pred_output = Dense(1)(out_hidden)
    if Z_non_linear and (mode in ['intercepts', 'glmm', 'survival']):
        Z_nll_inputs = []
        ls = []
        for k, q in enumerate(qs):
            l = int(q * Z_embed_dim_pct / 100.0)
            Z_embed = Embedding(q, l, input_length=1, name='Z_embed' + str(k))(Z_inputs[k])
            Z_embed = Reshape(target_shape=(l, ))(Z_embed)
            Z_nll_inputs.append(Z_embed)
            ls.append(l)
    elif mode == 'spatial_embedded':
        Z_embed = add_layers_functional(Z_inputs[0], spatial_embed_neurons, dropout=None, activation='relu', input_dim=2)
        Z_nll_inputs = [Z_embed]
        ls = [spatial_embed_neurons[-1]]
        Z_non_linear = True
    else:
        Z_nll_inputs = Z_inputs
        ls = None
    sig2bs_init = np.ones(n_sig2bs_init, dtype=np.float32)
    rhos_init = np.zeros(len(est_cors), dtype=np.float32)
    weibull_init = np.ones(2, dtype=np.float32)
    nll = COPNLL(mode, 1.0, sig2bs_init, rhos_init, weibull_init, est_cors, Z_non_linear, dmatrix_tf, fit_marginal)(
        y_true_input, y_pred_output, Z_nll_inputs)
    model = Model(inputs=[X_input, y_true_input] + Z_inputs, outputs=nll)

    model.compile(optimizer='adam')

    patience = epochs if patience is None else patience
    if Z_non_linear and mode == 'intercepts':
        # in complex scenarios such as non-linear g(Z) consider training "more", until var components norm has converged
        # callbacks = [EarlyStoppingWithSigmasConvergence(patience=patience)]
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    else:
        callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    if log_params:
        callbacks.extend([LogEstParams(idx, 1), CSVLogger('res_params.csv', append=True)])
    if not Z_non_linear:
        X_train.sort_values(by=z_cols, inplace=True)
        y_train = y_train[X_train.index]
    if mode == 'spatial_embedded':
        X_train_z_cols = [X_train[['D1', 'D2']]]
        X_test_z_cols = [X_test[['D1', 'D2']]]
    else:
        X_train_z_cols = [X_train[z_col] for z_col in z_cols]
        X_test_z_cols = [X_test[z_col] for z_col in z_cols]
    history = model.fit([X_train[x_cols], y_train] + X_train_z_cols, None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose, shuffle=shuffle)
    nll_tr = model.evaluate([X_train[x_cols], y_train] + X_train_z_cols, verbose=verbose)
    nll_te = model.evaluate([X_test[x_cols], y_test] + X_test_z_cols, verbose=verbose)

    sig2e_est, sig2b_ests, rho_ests, weibull_ests = model.layers[-1].get_vars()
    if mode in ['spatial', 'spatial_embedded']:
        sig2b_spatial_ests = sig2b_ests
        sig2b_ests = []
    elif mode == 'spatial_and_categoricals':
        sig2b_spatial_ests = sig2b_ests[:2]
        sig2b_ests = sig2b_ests[2:]
    else:
        sig2b_spatial_ests = []
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols, verbose=verbose).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, q_spatial, sig2e_est, sig2b_ests, sig2b_spatial_ests,
                Z_non_linear, model, ls, mode, rho_ests, est_cors, dist_matrix, weibull_ests, sample_n_train,
                copula=True, marginal=fit_marginal)
    dummy_y_test = np.random.normal(size=y_test.shape)
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        if Z_non_linear or len(qs) > 1 or mode == 'spatial_and_categoricals':
            delta_loc = 0
            if mode == 'spatial_and_categoricals':
                delta_loc = 1
            Z_tests = []
            for k, q in enumerate(qs):
                Z_test = get_dummies(X_test['z' + str(k + delta_loc)], q)
                if Z_non_linear:
                    W_est = model.get_layer('Z_embed' + str(k)).get_weights()[0]
                    Z_test = Z_test @ W_est
                Z_tests.append(Z_test)
            if Z_non_linear:
                Z_test = np.hstack(Z_tests)
            else:
                Z_test = sparse.hstack(Z_tests)
            if mode == 'spatial_and_categoricals':
                Z_test = sparse.hstack([Z_test, get_dummies(X_test['z0'], q_spatial)])
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
        else:
            # if model input is that large, this 2nd call to predict may cause OOM due to GPU memory issues
            # if that is the case use tf.convert_to_tensor() explicitly with a call to model() without using predict() method
            # y_pred = model([tf.convert_to_tensor(X_test[x_cols]), tf.convert_to_tensor(dummy_y_test), tf.convert_to_tensor(X_test_z_cols[0])], training=False).numpy().reshape(
            #     X_test.shape[0]) + b_hat[X_test['z0']]
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) #+ b_hat[X_test['z0']]
        if mode == 'glmm':
            y_pred = np.exp(y_pred)/(1 + np.exp(y_pred))
    elif mode == 'slopes':
        q = qs[0]
        Z0 = get_dummies(X_test['z0'], q)
        t = X_test['t'].values
        N = X_test.shape[0]
        Z_list = [Z0]
        for k in range(1, len(sig2b_ests)):
            Z_list.append(sparse.spdiags(t ** k, 0, N, N) @ Z0)
        Z_test = sparse.hstack(Z_list)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + Z_test @ b_hat
    elif mode == 'spatial_embedded':
        last_layer = Model(inputs = model.input[2], outputs = model.layers[-2].output)
        gZ_test = last_layer.predict(X_test_z_cols, verbose=verbose)
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0]) + gZ_test @ b_hat
        sig2b_spatial_ests = np.concatenate([sig2b_spatial_ests, [np.nan]])
    elif mode == 'survival':
        y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols, verbose=verbose).reshape(
                X_test.shape[0])
        y_pred = y_pred + np.log(b_hat[X_test['z0']])
    return y_pred, (sig2e_est, list(sig2b_ests), list(sig2b_spatial_ests)), list(rho_ests), len(history.history['loss']), nll_tr, nll_te


def run_embeddings(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience,
        n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose=False):
    if mode == 'glmm':
        loss = 'binary_crossentropy'
        last_layer_activation = 'sigmoid'
    else:
        loss = 'mse'
        last_layer_activation = 'linear'
    embed_dim = 10

    X_input = Input(shape=(X_train[x_cols].shape[1],))
    Z_inputs = []
    embeds = []
    qs_list = list(qs)
    if q_spatial is not None:
        qs_list +=  [q_spatial]
    for q in qs_list:
        Z_input = Input(shape=(1,))
        embed = Embedding(q, embed_dim, input_length=1)(Z_input)
        embed = Reshape(target_shape=(embed_dim,))(embed)
        Z_inputs.append(Z_input)
        embeds.append(embed)
    concat = Concatenate()([X_input] + embeds)
    out_hidden = add_layers_functional(concat, n_neurons, dropout, activation, X_train[x_cols].shape[1] + embed_dim * len(qs_list))
    output = Dense(1, activation=last_layer_activation)(out_hidden)
    model = Model(inputs=[X_input] + Z_inputs, outputs=output)

    model.compile(loss=loss, optimizer='adam')

    callbacks = [EarlyStopping(
        monitor='val_loss', patience=epochs if patience is None else patience)]
    X_train_z_cols = [X_train[z_col] for z_col in X_train.columns[X_train.columns.str.startswith('z')]]
    X_test_z_cols = [X_test[z_col] for z_col in X_train.columns[X_train.columns.str.startswith('z')]]
    history = model.fit([X_train[x_cols]] + X_train_z_cols, y_train,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose)
    y_pred = model.predict([X_test[x_cols]] + X_test_z_cols, verbose=verbose
                           ).reshape(X_test.shape[0])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_rhos = [None for _ in range(len(est_cors))]
    return y_pred, (None, none_sigmas, none_sigmas_spatial), none_rhos, len(history.history['loss']), None, None


def run_regression(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols,
        batch, epochs, patience, n_neurons, dropout, activation, reg_type,
        Z_non_linear, Z_embed_dim_pct, mode, n_sig2bs, n_sig2bs_spatial, est_cors,
        dist_matrix, time2measure_dict, spatial_embed_neurons, resolution, verbose,
        log_params, idx, shuffle, fit_marginal, b_true):
    start = time.time()
    if reg_type == 'ohe':
        y_pred, sigmas, rhos, n_epochs, nll_tr, nll_te = run_reg_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    elif reg_type == 'lmmnn':
        y_pred, sigmas, rhos, n_epochs, nll_tr, nll_te = run_lmmnn(
            X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode,
            n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons, verbose,
            Z_non_linear, Z_embed_dim_pct, log_params, idx, shuffle, b_true=b_true)
    elif reg_type == 'copnn':
        y_pred, sigmas, rhos, n_epochs, nll_tr, nll_te = run_copnn(
            X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode,
            n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons, fit_marginal, verbose,
            Z_non_linear, Z_embed_dim_pct, log_params, idx, shuffle, b_true=b_true)
    elif reg_type == 'ignore':
        y_pred, sigmas, rhos, n_epochs, nll_tr, nll_te = run_reg_ohe_or_ignore(
            X_train, X_test, y_train, y_test, qs, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose, ignore_RE=True)
    elif reg_type == 'embed':
        y_pred, sigmas, rhos, n_epochs, nll_tr, nll_te = run_embeddings(
            X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch, epochs, patience,
            n_neurons, dropout, activation, mode, n_sig2bs, n_sig2bs_spatial, est_cors, verbose)
    else:
        raise ValueError(reg_type + 'is an unknown reg_type')
    end = time.time()
    K.clear_session()
    gc.collect()
    if mode == 'glmm':
        metric = roc_auc_score(y_test, y_pred)
    else:
        metric = np.mean((y_pred - y_test)**2)
    return RegResult(metric, sigmas, rhos, nll_tr, nll_te, n_epochs, end - start)
