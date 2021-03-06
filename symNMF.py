from itertools import cycle
import warnings
from functools import partial
import itertools
import multiprocessing

import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_non_negative
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed
from bayes_opt import BayesianOptimization


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


def initialize_UV(X, n_components, init=None, eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = W * W.T
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: None.
        Valid options:
        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    eps : float
        Truncate all values less then this in output to zero.
    random_state : int, RandomState instance, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape
    assert n_samples == n_features, f'X must be symmetric, has shape {X.shape}'

    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))

    if init is None:
        if n_components <= n_samples:
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype,
                                                            copy=False)
        np.abs(W, out=W)
    else:

        # NNDSVD initialization
        # If A is a symmetric matrix, the singular values are the absolute
        # values of the eigenvalues of A and the columns of U=V are the
        # eigenvectors of A
        S, U = scipy.sparse.linalg.eigsh(X,
                                         k=n_components,
                                         which='LM',
                                         return_eigenvectors=True)
        S = np.abs(S)
        sort_idxs = np.argsort(S)[::-1]
        S = S[sort_idxs]
        U = U[:, sort_idxs]
        W = np.zeros_like(U)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])

        for j in range(1, n_components):
            x = U[:, j]

            # extract positive and negative parts of column vectors
            x_p = np.maximum(x, 0)
            x_n = np.abs(np.minimum(x, 0))

            # and their norms
            x_p_nrm = scipy.linalg.norm(x_p)
            x_n_nrm = scipy.linalg.norm(x_n)

            m_p, m_n = x_p_nrm * x_p_nrm, x_n_nrm * x_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u

        W[W < eps] = 0

        if init == "nndsvd":
            pass
        elif init == "nndsvda":
            avg = X.mean()
            W[W == 0] = avg
        elif init == "nndsvdar":
            rng = check_random_state(random_state)
            avg = X.mean()
            W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        else:
            raise ValueError(
                'Invalid init parameter: got %r instead of one of %r' %
                (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))
    H = W.copy()
    return W, H


class SymNMF:

    def __init__(self,
                 n_components,
                 max_iter=200,
                 tol=1E-4,
                 alpha=0.,
                 l1_ratio=0.5,
                 init=None,
                 random_state=None,
                 warm_start_ab=True,
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
        self.init = init
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
                    random_state=self.random_state,
                    init=self.init)
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
            init=None,
            random_state=None):
    n, m = Y.shape
    assert n == m, f'Input matrix of dimension {Y.shape} is not square'
    l1 = alpha * l1_ratio
    l2 = alpha - l1
    if A is None or B is None:
        A, B = initialize_UV(Y, J, init=init, random_state=random_state)
        init_err = nmf_err(Y, A, B)
    else:
        C, D = initialize_UV(Y, J, init=init, random_state=random_state)
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
               warm_start_ab=True,
               warm_start_lmda=True,
               outer_max_iter=20,
               outer_tol1=5E-2,
               outer_tol2=0.25,
               init=None,
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
    u, v = initialize_UV(yhat, J, init=init, random_state=random_state) if (warm_start_ab or warm_start_lmda) else (None, None)
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
                       init=init,
                       **kwargs)
        y_nmf = np.dot(u, v.T)
        new_vals = y_nmf[nanmask]
        scaled_nan_diff = mean_squared_error(old_vals, new_vals) / nanmean
        if scaled_nan_diff == 0:
            break
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


def multigen(gen_func):
    # from https://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs

        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen


@multigen
def repeat_replicate_pair_generator(reps):
    return itertools.permutations(reps, 2)


def symnmf_xval_neg_mse(train_test_gen,
                        n_components,
                        alphae,
                        l1_ratio,
                        n_jobs=1,
                        random_state=None,
                        **nmf_kwargs):
    alpha = 10 ** alphae
    nmf_kwargs.update(dict(n_components=int(round(n_components)),
                           alpha=alpha,
                           l1_ratio=l1_ratio))
    mse = symnmf_xval(train_test_gen,
                      n_jobs=n_jobs,
                      random_state=random_state,
                      **nmf_kwargs)
    return -1 * mse  # since we'll be maximizing


def symnmf_xval(train_test_gen, n_jobs=1, **nmf_kwargs):
    def _parallel_split_mse(train, test):
        nmf = SymNMF(**nmf_kwargs).fit(train)
        recons = np.dot(nmf.U, nmf.V.T)
        nonan = ~np.isnan(test)
        mse = mean_squared_error(recons[nonan], test[nonan])
        return mse
    if n_jobs == 1:
        mses = [_parallel_split_mse(train, test) for train, test in train_test_gen]
    else:
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        mses = Parallel(n_jobs=n_jobs)(delayed(_parallel_split_mse)(train, test) for train, test in train_test_gen)
    return np.mean(mses)


def select_model(mean_x,
                 train_test_gen,
                 min_rank,
                 max_rank,
                 min_l1_ratio=1e-3,
                 max_l1_ratio=1,
                 min_alphae=None,
                 max_alphae=None,
                 alpha_eps=1E-5,
                 init_points=20,
                 n_iter=100,
                 n_jobs=1,
                 random_state=None,
                 **nmf_kwargs):
    rank_bounds = (min_rank, max_rank)
    l1_ratio_bounds = (min_l1_ratio, max_l1_ratio)
    if max_alphae is None:
        max_alpha = estimate_max_l1(mean_x, rank_bounds[0], l1_ratio_bounds[0])
        max_alphae = np.log10(max_alpha)
    if min_alphae is None:
        min_alpha = max_alphae * alpha_eps
        min_alphae = np.log10(min_alpha)
    else:
        min_alphae = min(min_alphae, max_alphae)
    alphae_bounds = (min_alphae, max_alphae)
    pbounds = {'n_components': rank_bounds,
               'alphae': alphae_bounds,
               'l1_ratio': l1_ratio_bounds}
    optimizer = BayesianOptimization(
        f=partial(symnmf_xval_neg_mse,
                  train_test_gen=train_test_gen,
                  n_jobs=n_jobs,
                  random_state=random_state,
                  **nmf_kwargs),
        pbounds=pbounds,
        random_state=random_state,
    )
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )
    opt = optimizer.max['params']
    best_search_neg_mse = optimizer.max['target']
    best_model = SymNMF(n_components=int(round(opt['n_components'])),
                        alpha=10 ** opt['alphae'],
                        l1_ratio=opt['l1_ratio'],
                        **nmf_kwargs).fit(mean_x)
    return best_model, best_search_neg_mse
