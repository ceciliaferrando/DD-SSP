import math
import numpy as np
from scipy import sparse
import pandas as pd
import copy

import sys
sys.path.append('..')

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.special import softmax, expit
from scipy.optimize import minimize, fmin_bfgs, fmin_tnc
from scipy import sparse
import tensorflow as tf
from absl import flags
from datetime import datetime
from itertools import combinations
from tqdm.auto import tqdm

from private_pgm_local.src.mbi import Dataset, FactoredInference
from diffprivlib_main.diffprivlib_local import models as dp
from private_pgm_local.mechanisms import aim
from dpsynth.workload import Workload

from datetime import datetime
from itertools import combinations
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pdb
import itertools
import time
import argparse
import warnings

from dpquery_utils import PGMsynthesizer
from dpquery_chebyshev import Chebyshev, phi_logit


# GENOBJPERT UTILS

import numpy as np

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
    #b = np.random.multivariate_normal(np.zeros((b_var.shape[0], )), np.sqrt(b_var))

    return Delta, b


def dp_objective(theta, X, y, n, d, Delta, b):

    base_loss = LogisticRegressionObjective.loss(theta, X, y)
    regularizer = 1/n * 0   #assumed zero
    sec_term = Delta/(2*n) * np.dot(theta.T, theta)
    third_term = np.dot(b.T, theta)/n

    return base_loss + regularizer + sec_term + third_term


def dp_gradient(theta, X, y, n, d, Delta, b):
    # exponent = y * (X.dot(theta))
    # gradient_loss = - (np.transpose(X) @ (y / (1 + np.exp(exponent)))) / (
    #     X.shape[0])
    # base_gradient = np.sum(gradient_loss.T, axis=0)/gradient_loss.T.shape[0]

    base_gradient = LogisticRegressionObjective.gradient(theta, X, y)

    reg_term = 1/n * np.zeros((d,))  # Assumed zero
    second_term = Delta/n * theta
    third_term = b/n

    return base_gradient + reg_term + second_term + third_term

def get_dp_approx_ll(theta, yTX, XTXy2, a, b, c, n):
    dp_approx_ll = n * a + np.dot(theta, yTX) * b + np.dot(np.dot(theta, XTXy2), theta) * c
    return dp_approx_ll


def testPrivApproxSSLogReg(yTX, XTXy2, X_test, y_test, n, C):
    y_test = 2 * y_test - 1
    ch = Chebyshev(-6, 6, 3, phi_logit)
    a, b, c = ch.c
    alpha = 1 / (n * C)
    clf = DPApproxLL(ch, yTX, XTXy2, n, penalty='l2', alpha=alpha)
    clf.fit()
    y_pred = clf.predict(X_test, threshold=0.5)
    y_pred_proba = clf.predict_proba(X_test)
    DPapprox_accuracy = accuracy_score(y_test, y_pred)
    DPapprox_f1score = f1_score(y_test, y_pred)
    DPapprox_fpr, DPapprox_tpr, DPapprox_threshold = roc_curve(y_test, y_pred_proba)
    DPapprox_auc = auc(DPapprox_fpr, DPapprox_tpr)
    return (DPapprox_f1score, DPapprox_accuracy,
            DPapprox_fpr, DPapprox_tpr,
            DPapprox_threshold, DPapprox_auc)


def testApproxSSLogReg(X, y, X_test, y_test):
    y = 2 * y - 1
    y_test = 2 * y_test - 1
    C = 0.001
    alpha = 1 / (C * len(X))
    ch = Chebyshev(-6, 6, 3, phi_logit)
    clf = ApproxLL(ch, 'l2', alpha)
    clf.fit(X, y)
    # evaluate performance on test set
    y_pred_proba = clf.predict_proba(X_test)
    public_approx_fpr, public_approx_tpr, _ = roc_curve(y_test, y_pred_proba)
    public_approx_auc = auc(public_approx_fpr, public_approx_tpr)
    public_approx_accuracy = clf.accuracy(X_test, y_test, 0.5)
    return (_, public_approx_accuracy,
            public_approx_fpr, public_approx_tpr,
            _, public_approx_auc)


def testLogReg(synth_X, synth_y, X_test, y_test):
    lr = LogisticRegression(penalty=None, fit_intercept=False)
    lr.fit(synth_X, synth_y)
    pred_y, prob_y_1 = lr.predict(X_test), lr.predict_proba(X_test)[:, 1]
    f1score = f1_score(y_test, pred_y)
    accuracy = accuracy_score(y_test, pred_y)
    fpr, tpr, threshold = roc_curve(y_test, prob_y_1)  # check that it's the actual values not the thresholded
    roc_auc = auc(fpr, tpr)
    return (f1score, accuracy, fpr, tpr, threshold, roc_auc)


def testPrivLogReg(X, y, X_test, y_test, privatelr):
    privatelr.fit(X, y)
    pred_y, prob_y_1 = privatelr.predict(X_test), privatelr.predict_proba(X_test)[:, 1]
    f1score = f1_score(y_test, pred_y)
    accuracy = accuracy_score(y_test, pred_y)
    fpr, tpr, threshold = roc_curve(y_test, prob_y_1)
    roc_auc = auc(fpr, tpr)
    return (f1score, accuracy, fpr, tpr, threshold, roc_auc)


class ApproxLL():

    def __init__(self, ch, penalty, alpha):
        self.ch = ch
        self.penalty = penalty
        self.alpha = alpha
        self.theta = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        self.X_ = self.scaler.fit_transform(self.X_)
        self.optimize()
        return self

    def log_likelihood(self, theta):
        a, b = self.ch.a, self.ch.b
        term = 0
        for i in range(len(self.X_)):
            xithetayi = np.dot(self.X_[i, :], theta) * np.array(self.y_).flatten()[i]
            # clamp xithetayi
            #                 if xithetayi<a:
            #                     #print("<")
            #                     xithetayi=a
            #                 elif xithetayi>b:
            #                     #print(">")
            #                     xithetayi=b
            term_i_approx = self.ch.eval(xithetayi)
            term += term_i_approx
        term = 1 / self.X_.shape[0] * term
        return term

    def optimize(self):

        def l2_penalty(theta):
            return np.sum(theta ** 2)

        x0 = [.0] * len(self.X_[0])

        if self.penalty == 'none':
            res = minimize(lambda theta: -self.log_likelihood(theta),
                           x0,
                           method='L-BFGS-B',
                           options={'maxiter': 10000},
                           tol=0.00001)
            theta_star = res.x
            fun = res.fun

        elif self.penalty == 'l2':
            res = minimize(lambda theta: -self.log_likelihood(theta) + self.alpha * l2_penalty(theta),
                           x0,
                           method='L-BFGS-B',
                           options={'maxiter': 10000},
                           tol=0.00001)
            theta_star = res.x
            fun = res.fun - self.alpha * l2_penalty(theta_star)

        else:
            raise ValueError('Unknown penalty type, choose None or l2')

        self.theta = theta_star

    def predict_proba(self, X):
        z = np.dot(X, self.theta.T)
        y_pred_proba = expit(z)
        # y_pred_proba = 1 / (1 + np.exp(-np.dot(X, self.theta.T)))
        return (y_pred_proba)

    def predict(self, X, threshold=0.5):
        X = self.scaler.fit_transform(X)
        z = np.dot(X, self.theta.T)
        y_pred_proba = expit(z)
        y_pred = 2 * ((y_pred_proba >= threshold).astype(int)) - 1
        return (y_pred)

    def accuracy(self, X, y_test, threshold):
        X = self.scaler.fit_transform(X)
        y_pred = self.predict(X, threshold)
        accuracy_direct = accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return (accuracy)

    def auc(self, X, y_test):
        y_pred_proba = self.predict_proba(X)
        fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        return (roc_auc)


class DPApproxLL(ApproxLL):

    def __init__(self, ch, yTX, XTXy2, n, penalty, alpha):
        self.n = n
        self.ch = ch
        self.penalty = penalty
        self.alpha = alpha
        self.theta = None
        self.yTX = yTX
        self.XTXy2 = XTXy2

    def fit(self):
        self.optimize()
        return self

    def log_likelihood(self, theta):
        a, b, c = self.ch.c
        term = get_dp_approx_ll(theta, self.yTX, self.XTXy2, a, b, c, self.n)
        term = 1 / self.n * term
        return term

    def optimize(self):

        def l2_penalty(theta):
            return np.sum(theta ** 2)

        x0 = [.0] * len(self.yTX)

        if self.penalty == 'none':
            res = minimize(lambda theta: -self.log_likelihood(theta),
                           x0,
                           method='L-BFGS-B',
                           options={'maxiter': 10000},
                           tol=0.00001)
            theta_star = res.x
            fun = res.fun

        elif self.penalty == 'l2':
            res = minimize(lambda theta: -self.log_likelihood(theta) + self.alpha * l2_penalty(theta),
                           x0,
                           method='L-BFGS-B',
                           options={'maxiter': 10000},
                           tol=0.00001)
            theta_star = res.x
            fun = res.fun - self.alpha * l2_penalty(theta_star)

        else:
            raise ValueError('Unknown penalty type, choose None or l2')

        self.theta = theta_star

    def predict(self, X, threshold=0.5):
        # X = self.scaler.fit_transform(X)
        z = np.dot(X, self.theta.T)
        y_pred_proba = expit(z)
        y_pred = 2 * ((y_pred_proba >= threshold).astype(int)) - 1
        return (y_pred)