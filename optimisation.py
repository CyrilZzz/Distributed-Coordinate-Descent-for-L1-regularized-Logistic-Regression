import numpy as np
from pyspark.ml.linalg import Vectors

def t(x, a):
    return np.sign(x) * np.max(abs(x) - a, 0)

def coordinate_descent(iterator,w,z,beta,lmbd):
    delta_beta = np.zeros(len(beta))
    for row in iterator:
        x_j = Vectors.dense([row[i] for i in range(len(row)-1)])  # -1 not take the feature_id
        q = z - np.dot(delta_beta,x_j) + (beta[row[-1]] + delta_beta[row[-1]])*x_j  # wrong formula for now
        delta_beta[row[-1]] = t(np.sum(w * x_j * q), lmbd) / np.dot(w, x_j**2) - beta[row[-1]]
    yield delta_beta