import numpy as np

def sigmoid(x, beta):
    """x est une observation. Bêta et x vivent dans R^p"""
    return 1 / (1 + np.exp(-np.dot(beta, x)))


def update_vector():
    pass


def t(x, a):
    return np.sign(x) * max(abs(x) - a, 0)


def w(x, beta):
    p_i = sigmoid(beta, x)
    return p_i * (1 - p_i)


def z(y, x, beta):
    num = (y + 1) / 2 - sigmoid(x, beta)
    den = w(x, beta)
    return num / den


def q():
    return z - np.dot(delta_beta, x) + (beta + delta_beta) * x_i_j


def update_coordinate(matrice_x, y, beta, j, lmbd):
    """matrice_x est la matrice des observations"""
    w = w(x, beta_j)
    u = np.sum(w * x * q)
    delta_beta_star = t(u, lmbd) / np.dot(w, x **2) - beta_j

    return delta_beta_star
