import numpy as np
from tqdm import tqdm as tqdm
from itertools import product, combinations
from copy import deepcopy
from scipy.sparse.linalg import eigs
from numpy.linalg import multi_dot

def get_phi_u(x, k, betas):
    """
    Compute feature vector of sequence x for Spectrum Kernel SP(k)
    :param x: string, DNA sequence
    :param k: int, length of k-mers
    :param betas: list, all combinations of k-mers drawn from 'A', 'C', 'G', 'T'
    :return: np.array, feature vector of x
    """
    phi_u = np.zeros(len(betas))
    for i in range(len(x) - k + 1):
        kmer = x[i:i + k]
        for i, b in enumerate(betas):
            phi_u[i] += (b == kmer)
    return phi_u

def get_spectrum_K(X, k):
    """
    Compute K(x, y) for each x, y in DNA sequences for Spectrum Kernel SP(k)
    :param: X: pd.DataFrame, features
    :param k: int, length of k-mers
    :return: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    betas = [''.join(c) for c in product('ACGT', repeat=k)]
    phi_u = []
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Computing feature vectors'):
        phi_u.append(get_phi_u(x, k, betas))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_u[i], phi_u[j])
                K[j, i] = K[i, j]
    K = K
    return K

def get_WD_d(x, y, d, L):
    """
    Compute, for two sequences x and y, K(x, y)
    :param x: string, DNA sequence
    :param y: string, DNA sequence
    :param d: int, maximal degree
    :param L: int, length of DNA sequences
    :return:
        - K(x, y): float
    """
    c_t = 0
    for k in range(1, d + 1):
        beta_k = 2 * (d - k + 1) / d / (d + 1)
        c_st = 0
        for l in range(1, L - k + 1):
            c_st += (x[l:l + k] == y[l:l + k])
        c_t += beta_k * c_st
    return c_t

def get_WD_K(X, d):
    """
    Compute K(x, y) for each x, y in DNA sequences for Weighted Degree Kernel (d)
    :param: X: pd.DataFrame, features
    :param d: int, maximal degree
    :return:
        - K: np.array, kernel
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        L = len(x)
        K[i, i] = L - 1 + (1 - d) / 3
        for j, y in enumerate(X.loc[:, 'seq']):
            if j > i:
                K[i, j] = get_WD_d(x, y, d, L)
                K[j, i] = K[i, j]
    return K

def select_method(X, method):
    """
    Compute the kernel corresponding to the input method

    - SP_k{x} : Spectrum kernel with x = int
    - WD_d{x} : Weight degree kernel with x = int
 
    :param X: pd.DataFrame, features
    :param method: string, method to apply for building the kernel
    :return: np.array, K
    """
    m = method.split('_')
    if method[:2] == 'SP':
        k = int(m[1][1:])
        K = get_spectrum_K(X, k)
    elif method[:2] == 'WD' and method[2] != 'S':
        print(m)
        d = int(m[1][1:])
        K = get_WD_K(X, d)
    else:
        NotImplementedError('Method not implemented. Please refer to the documentation for choosing among available methods')
    return K