import numpy as np

def sigmoid(x,beta):
    return 1 / (1 + np.exp(-np.dot(beta, x)))


def update_vector():
    pass


def t(x, a):
    return np.sign(x) * np.max(abs(x) - a, 0)


def w(x, beta):
    p_i = sigmoid(beta, x)
    return p_i * (1 - p_i)


def z(y, x, beta):
    num = (y + 1) / 2 - sigmoid(x, beta)
    den = w(x, beta)
    return num / den


def update_coordinate(x, y, beta_):
    beta_star = t(np.sum(w * x * q), lmbd) / np.dot(w, x **2) - beta_j




