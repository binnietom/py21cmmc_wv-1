import numpy as np
from numpy.linalg import slogdet, solve
from scipy.sparse import issparse


def logdet_block_matrix(S, blocklen=None):
    if type(S) == list:
        return np.sum([slogdet(s)[1] for s in S])
    elif issparse(S):
        return np.sum([slogdet(S[i * blocklen:(i + 1) * blocklen,
                               i * blocklen:(i + 1) * blocklen].toarray())[1] for i in
                       range(int(S.shape[0] / blocklen))])
    else:
        return np.sum([slogdet(S[i * blocklen:(i + 1) * blocklen,
                               i * blocklen:(i + 1) * blocklen])[1] for i in
                       range(int(S.shape[0] / blocklen))])


def solve_block_matrix(S, x, blocklen=None):
    if type(S)==list:
        bits = [solve(s, x[i*len(s):(i+1)*len(s)]) for i, s in enumerate(S)]
    elif issparse(S):
        bits = [solve(S[i * blocklen:(i + 1) * blocklen, i * blocklen:(i + 1) * blocklen].toarray(),
                      x[i * blocklen:(i + 1) * blocklen]) for i in range(int(S.shape[0] / blocklen))]
    else:
        bits = [solve(S[i * blocklen:(i + 1) * blocklen, i * blocklen:(i + 1) * blocklen],
                      x[i * blocklen:(i + 1) * blocklen]) for i in range(int(S.shape[0] / blocklen))]
    bits = np.array(bits).flatten()
    return bits


def lognormpdf(x, mu, cov, blocklen=None):
    """
    Calculate gaussian probability log-density of x, when x ~ N(mu,sigma), and S is sparse.

    Code adapted from https://stackoverflow.com/a/16654259
    """
    nx = len(x)
    norm_coeff = nx * np.log(2 * np.pi) + logdet_block_matrix(cov, blocklen)

    err = x - mu

    numerator = solve_block_matrix(cov, err, blocklen).T.dot(err)

    return -0.5 * (norm_coeff + numerator)
