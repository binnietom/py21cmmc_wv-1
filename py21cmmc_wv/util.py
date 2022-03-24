from numpy.linalg import slogdet, solve, LinAlgError
import numpy as np
import math
import warnings

from scipy.stats import multivariate_normal
from scipy.sparse.linalg import spsolve

def lognormpdf(x,mu,cov):
    """
    Calculate gaussian probability density of x, when x ~ N(mu,sigma)
    """
    L = 0
    for i in range(len(x)):
        nx = len(cov[i])
        norm_coeff = nx*math.log(2*math.pi)+slogdet(cov[i])[1]
        err = x[i]-mu[i]
        numerator = spsolve(cov[i], err).T.dot(err)
        L += -0.5*(norm_coeff+numerator)
    return L

def loaded_lognormpdf(x,mu,cov):
    """
    As lognormpdf but adapted to cater for loading cov matrix. (6192 = 49*129)
    Currently 1 chunk cov with 260x260 centres
    Within the nchunks loop:
    Loaded npy shape is (nchunks, kperpmod, kpar, centres, centres)
    Estimated shape is (kperpmod x kpar, centre, centres).
    """
    L = 0
    nx = len(x)
    for i in range(int(nx-len(cov[0,:,0,0]))): #(len(kperp_mod)-1)xlen(kparr)
        iwrap_kparr = int(i - np.floor(i/len(cov[0,:,0,0]))*len(cov[0,:,0,0]))
        iwrap_kperp = int(np.floor(i/len(cov[0,:,0,0])))
        cv = cov[iwrap_kperp,iwrap_kparr,:,:]
        err = x[i]-mu[i]
        #try:
        norm_coeff = nx*math.log(2*math.pi)+slogdet(cv)[1]
        #except:
        numerator = spsolve(cv, err).T.dot(err)
        L += -0.5*(norm_coeff+numerator)
    return L

def straightlognorm(x,mu,cov):
    """
    a least code check for when block diagonal causes issues (typically from zeros)
    """
    L = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            L += np.log(multivariate_normal(x[i], cov[i]).pdf(mu[i]))
    return L


def logdet_block_matrix(S):
    sm = 0
    for s in S:
        if not np.all(s==0): # if all s is zero, then there's no signal here at all... move on.
            try:
                sm += slogdet(s)[1]
            except LinAlgError as e:
                # If the log-det can't be found for this block, just ignore it and move on.
                # TODO: this is probably a bad idea!
                warnings.warn("log-determinant not working: %s"%s)

    return sm

def solve_block_matrix(S, x):
    bits = []
    inds = []
    for i, (s,xx) in enumerate(zip(S, x)):
        if not np.all(s == 0):  # if all s is zero, then there's no signal here at all... move on.
            try:
                sol = solve(s, xx)
                bits += [sol]
                inds += [i]
            except LinAlgError:
                # Sometimes, the covariance might be all zeros, or singular.
                # Then we just ignore those values of u (or kperp) and keep going.
                # TODO: this might not be a great idea.

                warnings.warn("solve didn't work for index %s" % i)

    bits = np.array(bits)
    return bits #, inds


def solve_block_matrix_T(S, x):
    matrix = solve_block_matrix(S, x)
    return solve_block_matrix[0].T

def lognormpdf_sparse(x, mu, cov):
    """
    Calculate gaussian probability log-density of x, when x ~ N(mu,cov), and cov is block diagonal.

    Code adapted from https://stackoverflow.com/a/16654259
    """

    nx = len(x)
    norm_coeff = nx * np.log(2 * np.pi) + logdet_block_matrix(cov)
    err = x - mu
    sol, inds = solve_block_matrix(cov, err)
    numerator = 0
    for i, (s, e) in enumerate(zip(sol, err[inds])):
        numerator += s.dot(e)
    return -0.5 * (norm_coeff + numerator)
