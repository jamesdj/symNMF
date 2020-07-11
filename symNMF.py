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
from bayes_opt import BayesianOptimization #, SequentialDomainReductionTransformer
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


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
    random_state = check_random_state(random_state)
    #x_mean = X.mean()
    #expon_mean = np.sqrt(x_mean / r)
    #U = random_state.exponential(scale=expon_mean, size=(n, r))
    #V = U.copy()
    #avg = np.sqrt(X.mean() / r)
    #U = avg * random_state.randn(n, r).astype(X.dtype,
    #                                          copy=False)
    #np.abs(U, out=U)
    w, v = scipy.sparse.linalg.eigsh(X, k=r)
    U = np.abs(v * np.sqrt(w))
    V = U.copy()
    return U, V


class SymNMF:

    def __init__(self,
                 n_components,
                 max_iter=200,
                 tol=1E-4,
                 alpha=0.,
                 l1_ratio=0.5,
                 random_state=None,
                 warm_start_ab=False,
                 warm_start_lmda=True,
                 outer_max_iter=200,
                 outer_tol1=5E-2,
                 outer_tol2=0.25,
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
        self.outer_tol1 = outer_tol1
        self.outer_tol2 = outer_tol2
        self.U = None
        self.V = None

    def fit(self, X, lmda=None):
        func = partial(symHALSnan,
                       warm_start_ab=self.warm_start_ab,
                       warm_start_lmda=self.warm_start_lmda,
                       outer_max_iter=self.outer_max_iter,
                       outer_tol1=self.outer_tol1,
                       outer_tol2=self.outer_tol2) if np.isnan(X).any() else symHALS
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
               outer_max_iter=20,
               outer_tol1=5E-2,
               outer_tol2=0.25,
               random_state=None,
               **kwargs):
    """
    Todo:
    - explore convergence criteria more, make sure it's robust
    - explore different initializations
    - why are warm starts not faster?
    - quality of solution and similarity of u and v with warm starts, not exact lambda
      - it seems the lambda decreases as we go, so we are ok; the first lambda is larger than we need
    """

    random_state = check_random_state(random_state)
    nanmask = np.isnan(Y)
    yhat = Y.copy()
    nanmean = np.nanmean(Y)
    yhat[nanmask] = nanmean
    old_vals = yhat[nanmask]
    u, v = initialize_UV(yhat, J, random_state=random_state) if (warm_start_ab or warm_start_lmda) else (None, None)
    lmda = compute_default_lmda(yhat, u, v) if warm_start_lmda else None
    if not warm_start_ab:
        u, v = None, None
    n_iter = 0
    kwargs.update({'lmda': lmda,
                   'A': u,
                   'B': v,
                   'random_state': random_state})
    scaled_nan_diffs = []
    while n_iter < outer_max_iter:
        u, v = symHALS(yhat,
                       J,
                       **kwargs)
        y_nmf = np.dot(u, v.T)
        new_vals = y_nmf[nanmask]
        scaled_nan_diff = mean_squared_error(old_vals, new_vals) / nanmean
        scaled_nan_diffs.append(scaled_nan_diff)
        if scaled_nan_diff < outer_tol1:
            if len(scaled_nan_diffs) > 1:
                orders_mag = np.abs(np.diff(np.log10(scaled_nan_diffs[-2:])))[0]
                if orders_mag < outer_tol2 * 5:
                    break
        yhat[nanmask] = new_vals
        old_vals = new_vals
        n_iter += 1
    if n_iter == outer_max_iter:
        warnings.warn("Maximum number of iterations %d reached. Increase it to"
                      " improve convergence." % outer_max_iter, ConvergenceWarning)
        print(f'Scaled nan diffs:\n{scaled_nan_diffs}')

    return u, v


def add_reflected_indices(fold):
    idxs1, idxs2 = [list(tup) for tup in fold]
    for idx1, idx2 in zip(*fold):
        if idx1 != idx2:
            idxs1.append(idx2)
            idxs2.append(idx1)
    return tuple([tuple(idxs1), tuple(idxs2)])


def symnmf_xval(x, n_folds=10, n_jobs=-1, n_reps=1, random_state=None, **nmf_kwargs):
    random_state = check_random_state(random_state)
    n, m = x.shape
    assert n == m, f'x must be symmetric, has shape {x.shape}'
    idxs = list(zip(*np.triu_indices(n)))
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    def parallel_symnmf_xval(fold):
        x_copy = x.copy()
        x_copy[fold] = np.nan
        nmf = SymNMF(**nmf_kwargs)
        nmf.fit(x_copy)
        pred = np.dot(nmf.U, nmf.V.T)
        nonan = ~np.isnan(x[fold])
        mse = mean_squared_error(x[fold][nonan], pred[fold][nonan])
        return mse

    rep_mses = []
    for _ in range(n_reps):
        folds = [tuple(zip(*fold)) for fold in shuffle_and_deal(idxs,
                                                                n_folds,
                                                                random_state=random_state)]
        folds = [add_reflected_indices(fold) for fold in folds]
        fold_mses = Parallel(n_jobs=n_jobs)(delayed(parallel_symnmf_xval)(fold) for fold in folds)
        rep_mses.append(np.mean(fold_mses))
    return np.mean(rep_mses)


def symnmf_xval_rank(x, ncs,
                     n_folds=10,
                     n_reps=1,
                     random_state=None,
                     n_jobs=-1,
                     verbose=False,
                     **nmf_kwargs):
    random_state = check_random_state(random_state)
    mses = []
    for nc in ncs:
        if verbose:
            print(nc)
        try:
            nmf_kwargs.update({'n_components': nc})
            mse = symnmf_xval(x,
                              n_folds=n_folds,
                              n_jobs=n_jobs,
                              n_reps=n_reps,
                              random_state=random_state,
                              **nmf_kwargs)
        except Exception as e:
            print(nc, e)
            raise e
            mse = np.nan
        mses.append(mse)
    return mses


def estimate_max_l1(x_orig, k, l1_ratio):
    if np.any(np.isnan(x_orig)):
        nmf = SymNMF(n_components=k).fit(x_orig)
        x = np.dot(nmf.U, nmf.V.T)
    else:
        x = x_orig
    n, m = x.shape
    z = np.zeros((n, k))
    max_err = nmf_err(x, z, z)
    x_mean = x.mean()
    expon_mean = np.sqrt(x_mean / k)
    return max_err / (n * k * expon_mean) / l1_ratio


def snmf_xval_neg_mse(x, k, alphae, l1_ratio, n_reps=1, n_folds=10, n_jobs=-1, random_state=None, **nmf_kwargs):
    alpha = 10 ** alphae
    nmf_kwargs.update(dict(n_components=int(round(k)), alpha=alpha, l1_ratio=l1_ratio))
    mse = symnmf_xval(x, n_folds=n_folds, n_jobs=n_jobs, n_reps=n_reps, random_state=random_state, **nmf_kwargs)
    return -1 * mse  # since we'll be maximizing


def select_model(x,
                 min_rank,
                 max_rank,
                 noise_var=None,
                 whitekernel=True,
                 whitekernel_noise_spread=1E5,
                 base_kernel=None,
                 min_l1_ratio=1e-3,
                 max_l1_ratio=1,
                 min_alphae=None,
                 max_alphae=None,
                 alpha_eps=1E-5,
                 init_points=20,
                 n_iter=100,
                 n_folds=10,
                 n_reps=1,
                 n_jobs=-1,
                 random_state=None,
                 n_init_fits=10,
                 n_final_fits=10,
                 **nmf_kwargs):
    k_bounds = (min_rank, max_rank)
    if noise_var is None:
        mid_k = int(round(np.mean([k_bounds])))
        neg_mses = [snmf_xval_neg_mse(x,
                                      mid_k,
                                      0,
                                      0,
                                      n_folds=n_folds,
                                      n_reps=n_reps,
                                      random_state=random_state,
                                      n_jobs=n_jobs,
                                      **nmf_kwargs) for _ in range(n_init_fits)]
        noise_var = np.std(neg_mses) ** 2
    l1_ratio_bounds = (min_l1_ratio, max_l1_ratio)
    if max_alphae is None:
        max_alpha = estimate_max_l1(x, k_bounds[0], l1_ratio_bounds[0])
        max_alphae = np.log10(max_alpha)
    if min_alphae is None:
        min_alpha = max_alpha * alpha_eps
        min_alphae = np.log10(min_alpha)
    else:
        min_alphae = min(min_alphae, max_alphae)
    alphae_bounds = (min_alphae, max_alphae)
    pbounds = {'k': k_bounds,
               'alphae': alphae_bounds,
               'l1_ratio': l1_ratio_bounds}

    # apparently only in repo, not conda version
    #bounds_transformer = SequentialDomainReductionTransformer()
    optimizer = BayesianOptimization(
        f=partial(snmf_xval_neg_mse,
                  x=x,
                  n_reps=n_reps,
                  n_folds=n_folds,
                  n_jobs=n_jobs,
                  random_state=random_state,
                  **nmf_kwargs),
        pbounds=pbounds,
        random_state=random_state,
        #bounds_transformer=bounds_transformer,
    )
    kernel = Matern(nu=2.5) if base_kernel is None else base_kernel
    if whitekernel:
        kernel = kernel + WhiteKernel(noise_level=noise_var,
                                      noise_level_bounds=(noise_var / whitekernel_noise_spread,
                                                          noise_var * whitekernel_noise_spread))
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
        kernel=kernel,
        alpha=(0 if whitekernel else noise_var)
    )
    opt = optimizer.max['params']
    best_search_neg_mse = optimizer.max['target']
    # Now fit multiple times and take model with lowest reconstruction error
    best_model = None
    best_mse = np.inf
    for _ in range(n_final_fits):
        nmf = SymNMF(n_components=int(round(opt['k'])),
                     alpha=10 ** opt['alphae'],
                     l1_ratio=opt['l1_ratio']).fit(x)
        nonan = ~np.isnan(x)
        pred = np.dot(nmf.U, nmf.V.T)
        mse = mean_squared_error(x[nonan], pred[nonan])
        if mse < best_mse:
            best_model = nmf
            best_mse = mse
    return best_model, best_search_neg_mse
