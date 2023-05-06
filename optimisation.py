import numpy as np

def t(x, a):
    return np.sign(x) * np.max(abs(x) - a, 0)

def coordinate_descent(iterator,w,z,num_features,beta,lmbd):
    delta_beta_m = np.zeros(num_features)
    for row in iterator:
        delta_beta_m[row[-1]] = t(np.sum(w * x * q), lmbd) / np.dot(w, x **2) - beta_j
    yield delta_beta_m