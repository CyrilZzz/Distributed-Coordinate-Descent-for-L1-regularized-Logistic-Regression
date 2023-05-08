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

def softplus(x):
    return np.log(1+np.exp(x))

def sigmoid2(x):
    return 1/(1+np.exp(x))

def objective_function(x,y,beta,lmbd):
    vsoftplus = softplus(- y * np.matmul(x,beta))
    return np.sum(vsoftplus) + lmbd*np.linalg.norm(beta,1)

def line_search(x, y, delta, beta, delta_beta,
                gamma, lmbd, sigma, b):
    """Creuser plus étape 1. de l'algo 3
    calculer de delta_L_beta dans la fonction
    calcul de la hessienne H_tilde : cf equation (3)"""
    # Step 2
    def eval_objective_function(alpha):
        return objective_function(x, y, beta + alpha*delta_beta, lmbd)
    alpha_init = minimize(eval_objective_function, np.array([0.5]), bounds=((delta, 1),)).x

    # Step 3 : Armijo Rule
    # gradient = np.sum(np.vectorize(sigmoid2(y*np.matmul(x,beta))))
    # premier_terme = np.dot(gradient, delta_beta)
    # deuxieme_terme = gamma * np.dot(delta_beta, np.matmul(H_tilde, delta_beta))
    # troisieme_terme = lmbd * (np.linalg.norm(beta + delta_beta, 1) - np.linalg.norm(beta, 1))
    # D = premier_terme + deuxieme_terme + troisieme_terme
    # alpha = alpha_init
    # for j in range(10**5):
    #     alpha_candidat = alpha_init * b**j
    #     if f(beta + alpha_candidat * delta_beta) <= f(beta) + alpha_candidat * sigma * D:
    #         if alpha_candidat > alpha:
    #             alpha = alpha_candidat
    return alpha_init
