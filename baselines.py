import math
import numpy as np
import pandas as pd
import copy
import os
import sys
sys.path.append('..')

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.special import expit, logsumexp
from scipy.optimize import minimize, fmin_tnc
from itertools import combinations

from private_pgm_local.src.mbi import Dataset, FactoredInference
from private_pgm_local.mechanisms import aim
from private_pgm_local.src.mbi.workload import Workload

from utils import *



# linear regression

def AdaSSP_linear_regression(X, y, epsilon, delta, rho, bound_X, bound_y, bound_XTX, XTy_budget_split):
    """Returns DP linear regression model and metrics using AdaSSP. AdaSSP is described in Algorithm 2 of
        https://arxiv.org/pdf/1803.02596.pdf.

    Args:
        X: df feature vectors
        y: df of labels
        epsilon: model needs to meet (epsilon, delta)-DP.
        delta: model needs to meet (epsilon, delta)-DP.
        rho: failure probability, default of 0.05 is from original paper
        bound_X, bound y: bounds on the L2 sensitivity
        bound_XTX: bound on the sensitivity of XTX (is data is one hot encoded, XTX is sparser, sensitivity must be adapted)
        X_test, y_test: test data for evaluation

    Returns:
        theta_dp: regression coefficients
        mse_dp: mean squared error
        r2_dp: r2 score
    """

    n, d = X.shape

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y).flatten()

    eigen_min = max(0, np.amin(np.linalg.eigvals(XTX)))
    z = np.random.normal(0, 1, size=1)
    epsilon_eigen, delta_eigen = epsilon/3, delta/3
    sensitivity_eigen = np.sqrt(np.log(2 / delta_eigen)) / (epsilon_eigen)
    eigen_min_dp = max(0,
                       eigen_min + sensitivity_eigen * (bound_XTX) * z -
                       (bound_XTX) * np.log(6 / delta) / (epsilon / 3))
    lambda_dp = max(0,
                    np.sqrt(d * np.log(2 / delta_eigen) * np.log(2 * (d ** 2) / rho)) * (bound_XTX) /
                    (epsilon_eigen) - eigen_min_dp)

    epsilon_XTX, delta_XTX = epsilon * 2/3 * (1.-XTy_budget_split), delta * 2/3 * (1.-XTy_budget_split)
    sensitivity_XTX = np.sqrt(np.log(2 / delta_XTX)) / (epsilon_XTX)
    tri = np.triu(np.random.normal(0, 1, (d, d)))
    Zsym = tri + tri.T - np.diag(np.diag(tri))
    XTX_dp = XTX + sensitivity_XTX * (bound_XTX) * Zsym

    epsilon_XTy, delta_XTy = epsilon * 2/3 * (XTy_budget_split), delta * 2/3 * (XTy_budget_split)
    sensitivity_XTy = np.sqrt(np.log(2 / delta_XTy)) / (epsilon_XTy)
    z = np.random.normal(0, 1, size=(d,))
    XTy_dp = XTy + sensitivity_XTy * bound_X * bound_y * z
    XTX_dp_reg = XTX_dp + lambda_dp * np.eye(d)

    theta_dp = np.linalg.solve(XTX_dp_reg, XTy_dp)

    theta_dp = pd.DataFrame(theta_dp)
    theta_dp = theta_dp.set_index(X.columns)

    return theta_dp, XTX_dp_reg, XTy_dp

# logistic regression

class LogisticRegressionObjective():
    @staticmethod
    def sigmoid_v2(x, theta):
        z = np.dot(x, theta)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def hypothesis(theta, x):
        return LogisticRegressionObjective.sigmoid_v2(x, theta)

    @staticmethod
    def loss(theta, x, y):
        m = x.shape[0]
        h = LogisticRegressionObjective.hypothesis(theta, x)
        return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    @staticmethod
    def gradient(theta, x, y):
        m = x.shape[0]
        h = LogisticRegressionObjective.hypothesis(theta, x)
        return (1 / m) * np.dot(x.T, (h - y))

def genobjpert_get_params(X, epsilon, delta, lmda, zeta):

    n, d = X.shape

    delta_thrs = 2*lmda/epsilon
    Delta = delta_thrs
    b_var = zeta**2 * (8 * np.log(2/delta) + 4*epsilon) / (epsilon**2) * np.eye(d)
    b = np.random.multivariate_normal(np.zeros((b_var.shape[0], )), b_var)

    return Delta, b

def dp_objective(theta, X, y, n, d, Delta, b):

    base_loss = LogisticRegressionObjective.loss(theta, X, y)
    regularizer = 1/n * 0   #assumed zero
    sec_term = Delta/(2*n) * np.dot(theta.T, theta)
    third_term = np.dot(b.T, theta)/n

    return base_loss + regularizer + sec_term + third_term

def dp_gradient(theta, X, y, n, d, Delta, b):
    base_gradient = LogisticRegressionObjective.gradient(theta, X, y)

    reg_term = 1/n * np.zeros((d,))  # Assumed zero
    second_term = Delta/n * theta
    third_term = b/n

    return base_gradient + reg_term + second_term + third_term

def objective_perturbation_method(X, y, epsilon, delta, bound_X, bound_y, all_columns):

    n, d = X.shape

    max_row_norm = bound_X
    zeta = max_row_norm
    lmda = smooth_const = max_row_norm ** 2 / 4

    def sigmoid_v2(x, theta):
        z = np.dot(x, theta)
        return 1 / (1 + np.exp(-z))

    def hypothesis(theta, x):
        return sigmoid_v2(x, theta)

    def cost_function(theta, x, y):
        m = x.shape[0]
        h = hypothesis(theta, x)
        return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient_fun(theta, x, y):
        m = x.shape[0]
        h = hypothesis(theta, x)
        return (1 / m) * np.dot(x.T, (h - y))

    # Run the object perturbation
    Delta, b = genobjpert_get_params(X.to_numpy(), epsilon, delta, lmda, zeta)

    # Run the iteration
    # fmin_tnc returns the convergence message as third argument in its output
    # 0 means local minimium reached, 1 and 2 means convergence by function value or theta value
    # 3 is maximum number of iterations reached, 4 linear search failed
    finish_opt = False
    patience = 5

    while not finish_opt and patience > 0:

        theta0 = np.random.normal(loc=0, scale=0.01, size=X.shape[1]).reshape(-1, )
        theta_opt = fmin_tnc(func=dp_objective, x0=theta0, fprime=dp_gradient, maxfun=10000, disp=0,
                             args=(X.to_numpy(), y.to_numpy().reshape(-1, ), n, d, Delta, b))

        theta_final, n_it_run, final_message = theta_opt

        if final_message in [0, 1, 2]:
            finish_opt = True
        else:
            patience -= 1

    theta_final = pd.DataFrame(theta_final, index=X.columns)
    theta_final = theta_final.reindex(index=all_columns, fill_value=0)

    return theta_final