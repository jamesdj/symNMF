import os
from functools import partial
from itertools import cycle
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
import numpy.ma as ma
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed


def shuffle_and_deal(cards, n_hands, random_state=None):
    random_state = check_random_state(random_state)
    shuffled = random_state.permutation(cards)
    hand_idxs = range(n_hands)
    hands = [list() for i in hand_idxs]
    cyc = cycle(hand_idxs)
    for card in shuffled:
        hand = hands[next(cyc)]
        hand.append(card)
    return hands


def emsvd(Y, k=None, tol=1E-3, max_iter=None):
    """
    From https://stackoverflow.com/questions/35577553/how-to-fill-nan-values-in-numeric-array-to-apply-svd

    Approximate SVD on data with missing values via expectation-maximization

    Inputs:
    -----------
    Y:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf
    k:          number of singular values/vectors to find (default: k=ndim)
    tol:        convergence tolerance on change in trace norm
    max_iter:    maximum number of EM steps to perform (default: no limit)

    Returns:
    -----------
    Y_hat:      (nobs, ndim) reconstructed data matrix
    mu_hat:     (ndim,) estimated column means for reconstructed data
    U, s, Vt:   singular values and vectors (see np.linalg.svd and
                scipy.sparse.linalg.svds for details)
    """
    if k is None:
        svdmethod = partial(np.linalg.svd, full_matrices=False)
    else:
        svdmethod = partial(svds, k=k)
    if max_iter is None:
        max_iter = np.inf
    # initialize the missing values to their respective column means
    mu_hat = np.nanmean(Y, axis=0, keepdims=1)
    valid = np.isfinite(Y)
    Y_hat = np.where(valid, Y, mu_hat)
    halt = False
    ii = 1
    v_prev = 0
    while not halt:
        # SVD on filled-in data
        U, s, Vt = svdmethod(Y_hat - mu_hat)
        # impute missing values
        Y_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]
        # update bias parameter
        mu_hat = Y_hat.mean(axis=0, keepdims=1)
        # test convergence using relative change in trace norm
        v = s.sum()
        if ii >= max_iter or ((v - v_prev) / v_prev) < tol:
            halt = True
        ii += 1
        v_prev = v

    return Y_hat, mu_hat, U, s, Vt


def ma_frob(X):
    return ma.sqrt(ma.sum(X ** 2))


def nmf_err(X, U, V):
    return 0.5 * (ma_frob(ma.subtract(X, ma.dot(U, V.T))) ** 2) / (ma_frob(X) ** 2)


def compute_default_lmda(x, U, V):
    x_svd = emsvd(x)[0] if x.mask.sum() > 0 else x
    svs = np.linalg.svd(x_svd, compute_uv=False)
    max_sv = svs[0]
    min_sv = svs[-1]
    lmda = 1.01 * (0.5 * (max_sv + ma_frob(ma.subtract(x, ma.dot(U, V.T))) - min_sv))
    return lmda


def initialize_UV(X, r, random_state=None):
    n, m = X.shape
    assert n == m, f'X must be symmetric, has shape {X.shape}'
    x_mean = X.mean()
    random_state = check_random_state(random_state)
    expon_mean = np.sqrt(x_mean / r)
    U = random_state.exponential(scale=expon_mean, size=(n, r))
    V = U.copy()
    return U, V


class SymNMF:

    def __init__(self,
                 n_components,
                 max_iter=200,
                 tol=1E-4,
                 alpha=0,
                 l1_ratio=0.5,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = check_random_state(random_state)
        self.U = None
        self.V = None

    def fit(self, X, lmda=None):
        U, V = symHALS(X, self.n_components,
                       max_iter=self.max_iter,
                       tol=self.tol,
                       lmda=lmda,
                       alpha=self.alpha,
                       l1_ratio=self.l1_ratio,
                       random_state=self.random_state)
        self.U = U
        self.V = V
        return self


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def symHALS(Y_orig, J, max_iter=200, tol=1E-4, lmda=None, alpha=0, l1_ratio=0.5, random_state=None):
    l1 = alpha * l1_ratio
    l2 = alpha - l1
    Y = ma.masked_invalid(Y_orig)
    A, B = initialize_UV(Y, J, random_state=random_state)
    if lmda is None:
        lmda = compute_default_lmda(Y, A, B)
    init_err = nmf_err(Y, A, B)
    err = float(init_err)
    n_iter = 0
    n, m = Y.shape
    n, J = A.shape
    S = np.zeros((m, m))
    S[kth_diag_indices(S, 1)] += 0.5
    S[kth_diag_indices(S, -1)] += 0.5
    S[0, 1] = 1
    S[n - 1, n - 2] = 1
    lmda_i = np.eye(n) * lmda
    Y_plus_lmda_i = Y + lmda_i
    while n_iter < max_iter:
        W = ma.dot(Y_plus_lmda_i.T, A)
        V = ma.dot(A.T, A)
        for j in range(J):
            foo = B[:, j] * V[j, j] + W[:, j] - ma.dot(B, V[:, j])
            B[:, j] = (foo
                       - lmda *
                       - l1 * np.ones(m)
                       + l2 * ma.dot(S, B[:, j])).clip(0, np.inf) / (V[j, j] + l2 + lmda)
        P = ma.dot(Y_plus_lmda_i, B)
        Q = ma.dot(B.T, B)
        for j in range(J):
            bar = A[:, j] * Q[j, j] + P[:, j] - ma.dot(A, Q[:, j])
            A[:, j] = (bar
                       - l1 * np.ones(m)
                       + l2 * ma.dot(S, A[:, j])).clip(0, np.inf) / (Q[j, j] + l2 + lmda)
        new_err = nmf_err(Y, A, B)
        #err_diff = err - new_err
        #if err_diff < tol:
        if (err - new_err) / init_err < tol:
            break
        err = new_err
        n_iter += 1
    if n_iter == max_iter:
        warnings.warn("Maximum number of iterations %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)
    return A, B


def add_reflected_indices(fold):
    idxs1, idxs2 = [list(tup) for tup in fold]
    for idx1, idx2 in zip(*fold):
        if idx1 != idx2:
            idxs1.append(idx2)
            idxs2.append(idx1)
    return tuple([tuple(idxs1), tuple(idxs2)])


def symnmf_xval(x, n_folds=10, n_jobs=-1, random_state=None, **nmf_kwargs):
    random_state = check_random_state(random_state)
    n, m = x.shape
    assert n == m, f'x must be symmetric, has shape {x.shape}'
    idxs = list(zip(*np.triu_indices(n)))
    folds = [tuple(zip(*fold)) for fold in shuffle_and_deal(idxs,
                                                            n_folds,
                                                            random_state=random_state)]
    folds = [add_reflected_indices(fold) for fold in folds]

    def parallel_nmf_xval(fold):
        x_copy = x.copy()
        x_copy[fold] = np.nan
        nmf = SymNMF(**nmf_kwargs)
        nmf.fit(x_copy)
        pred = np.dot(nmf.U, nmf.V.T)
        nonan = ~np.isnan(x[fold])
        mse = mean_squared_error(x[fold][nonan], pred[fold][nonan])
        return mse

    if n_jobs == -1:
        n_jobs = os.cpu_count()
    mses = Parallel(n_jobs=n_jobs)(delayed(parallel_nmf_xval)(fold) for fold in folds)
    return mses


def symnmf_xval_rank(x, ncs, n_folds=10, n_reps=1, random_state=None, n_jobs=-1, **nmf_kwargs):
    random_state = check_random_state(random_state)
    msess = []
    for nc in ncs:
        print(nc)
        reps_mses = []
        for r in range(n_reps):
            try:
                mses = symnmf_xval(x,
                                   n_folds=n_folds,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   n_components=nc,
                                   **nmf_kwargs)
            except Exception as e:
                print(nc, e)
                raise e
                mses = [np.nan] * n_folds
            reps_mses.extend(mses)
        msess.append(reps_mses)
    return msess



"""
# https://github.com/kimjingu/nonnegfac-python
from nonnegfac.nnls import nnlsm_blockpivot

def symANLS(X, r, max_iter=200, tol=1E-8, random_state=None):
    # From Zhu et al. 2018 "Dropping Symmetry for Fast Symmetric Nonnegative Matrix Factorization"
    
    U, V = initialize_UV(X, r, random_state=random_state)
    lmda = compute_default_lmda(X, U, V)
    err = nmf_err(X, U, V)
    n, r = V.shape
    i = 0
    while i <= max_iter:
        UT, ut_info = nnlsm_blockpivot(np.concatenate([V, np.sqrt(lmda) * np.eye(r)]), 
                              np.concatenate([X.T, np.sqrt(lmda) * V.T]), 
                              False, 
                              U.T)
        U = UT.T
        VT, vt_info = nnlsm_blockpivot(np.concatenate([U, np.sqrt(lmda) * np.eye(r)]),
                              np.concatenate([X, np.sqrt(lmda) * U.T]), 
                              False, 
                              V.T)
        V = VT.T
        new_err = nmf_err(X, U, V)
        err_diff = err - new_err
        err = new_err
        if err_diff < tol:
            break
        i += 1
    return U, V
"""

