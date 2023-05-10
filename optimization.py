import numpy as np
from pyspark.ml.linalg import Vectors
from scipy.optimize import minimize

def t(x, a):
    return np.sign(x) * np.max(abs(x) - a, 0)

# Solving M sub-problems
def coordinate_descent(partition, x, w, z, beta, lmbd):
    delta_beta = np.zeros(len(beta))
    for row in partition:
        x_j = Vectors.dense(row[:-1])  # Remove the feature_id
        q = z - np.matmul(x, delta_beta) + beta[row[-1]]*x_j
        delta_beta[row[-1]] = t(np.sum(w * x_j * q), lmbd) / np.dot(w, np.power(x_j, 2)) - beta[row[-1]]
    yield delta_beta

def softplus(x):
    return np.log(1 + np.exp(x))

def sigmoid2(x):
    return 1 / (1 + np.exp(x))

def objective_function(x, y, beta, lmbd):
    vsoftplus = softplus(-y * np.matmul(x, beta))
    return np.sum(vsoftplus) + lmbd * np.linalg.norm(beta, 1)

def line_search(x, y, delta, beta, delta_beta,
                gamma, lmbd, sigma, b):
    def eval_objective_function(alpha):
        return objective_function(x, y, beta + alpha * delta_beta, lmbd)
    # Initialize the sequence (step 2 from algorithm 3 in the original paper)
    alpha_init = minimize(eval_objective_function, np.array([0.5]), bounds=((delta, 1),)).x

    # Armijo Rule
    gradient = np.sum(sigmoid2(y*np.matmul(x, beta))) * beta
    D = np.dot(gradient, delta_beta) + lmbd * (np.linalg.norm(
        beta + delta_beta, 1) - np.linalg.norm(beta, 1))
    alpha = alpha_init
    j = 0
    while eval_objective_function(
            alpha) > eval_objective_function(0) + alpha * sigma * D:
        j += 1
        alpha = alpha_init * b**j
    return alpha
