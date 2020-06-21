import os
from itertools import cycle
import warnings
from functools import partial

import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning
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
                 random_state=None,
                 warm_start_ab=False,
                 warm_start_lmda=True,
                 outer_max_iter=200,
                 outer_tol=1E-2,
                 ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = check_random_state(random_state)
        self.warm_start_ab = warm_start_ab
        self.warm_start_lmda = warm_start_lmda
        self.outer_max_iter = outer_max_iter
        self.outer_tol = outer_tol
        self.U = None
        self.V = None

    def fit(self, X, lmda=None):
        func = partial(symHALSnan,
                       warm_start_ab=self.warm_start_ab,
                       warm_start_lmda=self.warm_start_lmda,
                       outer_max_iter=self.outer_max_iter,
                       outer_tol=self.outer_tol) if np.isnan(X).any() else symHALS
        U, V = func(X, self.n_components,
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


def nmf_err(X, U, V):
    UV = np.dot(U, V.T)
    diff = np.subtract(X, UV)
    diff_norm = np.linalg.norm(diff)
    sq_diff_norm = diff_norm ** 2
    err = 0.5 * sq_diff_norm
    return err


def compute_default_lmda(x, U, V):
    # slower but maybe more stable?
    # svs = np.linalg.svd(x, compute_uv=False, hermitian=True)
    # max_sv = svs[0]
    # min_sv = svs[-1]

    # max_sv = scipy.sparse.linalg.svds(x, k=1, which='LM', return_singular_vectors=False)[0]
    # min_sv = scipy.sparse.linalg.svds(x, k=1, which='SM', return_singular_vectors=False)[0]

    # slightly faster, can do bc symmetric and so equivalent to singular vals
    max_sv = scipy.sparse.linalg.eigsh(x, k=1, which='LM', return_eigenvectors=False)[0]
    # trick to get small eigvals faster and with better convergence
    # see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    min_sv = scipy.sparse.linalg.eigsh(x, k=1, sigma=0, which='LM', return_eigenvectors=False)[0]
    lmda = 1.01 * (0.5 * (max_sv + np.linalg.norm(np.subtract(x, np.dot(U, V.T))) - min_sv))
    return lmda


def make_smoothing_matrix(m):
    S = np.zeros((m, m))
    S[kth_diag_indices(S, 1)] += 0.5
    S[kth_diag_indices(S, -1)] += 0.5
    S[0, 1] = 1
    S[m - 1, m - 2] = 1  # would be n, but since symmetric
    return S


def symHALS(Y,
            J,
            A=None,
            B=None,
            max_iter=200,
            tol=1E-4,
            lmda=None,
            alpha=0,
            l1_ratio=0.5,
            random_state=None):
    n, m = Y.shape
    assert n == m, f'Input matrix of dimension {Y.shape} is not square'
    l1 = alpha * l1_ratio
    l2 = alpha - l1
    if A is None or B is None:
        A, B = initialize_UV(Y, J, random_state=random_state)
        init_err = nmf_err(Y, A, B)
    else:
        C, D = initialize_UV(Y, J, random_state=random_state)
        init_err = nmf_err(Y, C, D)
    if lmda is None:
        lmda = compute_default_lmda(Y, A, B)
    err = float(init_err)
    S = make_smoothing_matrix(m)
    lmda_i = np.eye(n) * lmda
    Y_plus_lmda_i = Y + lmda_i

    def hals_update(A, B):
        W = np.dot(Y_plus_lmda_i.T, A)
        V = np.dot(A.T, A)
        for j in range(J):
            resid_term = B[:, j] * V[j, j] + W[:, j] - np.dot(B, V[:, j])
            B[:, j] = (resid_term
                       - l1 * np.ones(m)
                       + l2 * np.dot(S, B[:, j])).clip(0, np.inf) / (V[j, j] + l2 + lmda)
        return B

    n_iter = 0
    while n_iter < max_iter:
        B = hals_update(A, B)
        A = hals_update(B, A)
        new_err = nmf_err(Y, A, B)
        rel_err = (err - new_err) / init_err
        if 0 <= rel_err < tol:
            break
        err = new_err
        n_iter += 1
    if n_iter == max_iter:
        warnings.warn("Maximum number of iterations %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)
    return A, B


def symHALSnan(Y,
               J,
               warm_start_ab=False,
               warm_start_lmda=True,
               outer_max_iter=200,
               outer_tol=1E-2,
               random_state=None,
               **kwargs):
    """
    Todo:
    - explore convergence criteria more, make sure it's robust
    - why are warm starts not faster?
    - quality of solution and similarity of u and v with warm starts, not exact lambda
      - it seems the lambda decreases as we go, so we are ok; the first lambda is larger than we need
    """

    random_state = check_random_state(random_state)
    nanmask = np.isnan(Y)
    yhat = Y.copy()
    nanmean = np.nanmean(Y)
    yhat[nanmask] = nanmean
    first_vals = yhat[nanmask]
    old_vals = first_vals.copy()
    diff_ratio = 1
    u, v = initialize_UV(yhat, J, random_state=random_state) if (warm_start_ab or warm_start_lmda) else (None, None)
    lmda = compute_default_lmda(yhat, u, v) if warm_start_lmda else None
    if not warm_start_ab:
        u, v = None, None
    n_iter = 0
    kwargs.update({'lmda': lmda,
                   'A': u,
                   'B': v,
                   'random_state': random_state})
    while diff_ratio > outer_tol and n_iter < outer_max_iter:
        u, v = symHALS(yhat,
                       J,
                       **kwargs)
        y_nmf = np.dot(u, v.T)
        new_vals = y_nmf[nanmask]
        diff = np.mean((old_vals - new_vals) ** 2)
        orig_diff = np.mean((first_vals - new_vals) ** 2)
        diff_ratio = diff / orig_diff
        yhat[nanmask] = new_vals
        old_vals = new_vals
        n_iter += 1
    return u, v


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
                nmf_kwargs.update({'n_components': nc})
                mses = symnmf_xval(x,
                                   n_folds=n_folds,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   **nmf_kwargs)
            except Exception as e:
                print(nc, e)
                raise e
                mses = [np.nan] * n_folds
            reps_mses.extend(mses)
        msess.append(reps_mses)
    return msess

