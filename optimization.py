import numpy as np
from pyspark.ml.linalg import Vectors
from scipy.optimize import minimize

def t(x, a):
    return np.sign(x) * np.max(abs(x) - a, 0)

# Solving M sub-problems
def coordinate_descent(partition,x,w,z,beta,lmbd):
    delta_beta = np.zeros(len(beta))
    for row in partition:
        x_j = Vectors.dense(row[:-1])  # -1 not take the feature_id
        q = z - [np.dot(delta_beta,x_i) for x_i in x] + beta[row[-1]]*x_j  # to be checked
        delta_beta[row[-1]] = t(np.sum(w * x_j * q), lmbd) / np.dot(w, np.power(x_j, 2)) - beta[row[-1]]
    yield delta_beta


def line_search(delta, f, beta, delta_beta, delta_L_beta,
                gamma, H_tilde, lmbd, sigma, b):
    """Creuser plus étape 1. de l'algo 3"""
    alpha_init = minimize(f, np.array([0.5]), bounds=((delta, 1),))
    premier_terme = np.dot(delta_L_beta, delta_beta)
    deuxieme_terme = gamma * np.dot(delta_beta, np.matmul(H_tilde, delta_beta))
    troisieme_terme = lmbd * (np.linalg.norm(beta + delta_beta, 1) - np.linalg.norm(beta, 1))
    D = premier_terme + deuxieme_terme + troisieme_terme
    suite = []
    alpha = alpha_init
    for j in range(10**5):
        alpha_candidat = alpha_init * b**j
        if f(beta + alpha_candidat * delta_beta) <= f(beta) + alpha_candidat * sigma * D:
            if alpha_candidat > alpha:
                alpha = alpha_candidat
    return alpha



