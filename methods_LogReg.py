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

from utils_LogReg import *
from utils import *

def public_method(X, y, X_test, y_test, cols_to_dummy, attribute_dict, encoded_features, one_hot, approx):
    if one_hot:
        X_pub = normalize_minus1_1(X, attribute_dict, encoded_features)
        X_test_pub = normalize_minus1_1(X_test, attribute_dict, encoded_features)
    else:
        X_pub = copy.deepcopy(X)
        X_test_pub = copy.deepcopy(X_test)

    if approx:
        # use the same chebyshev approximation as DPQuerySS, for comparison
        (public_f1score, public_accuracy,
         public_fpr, public_tpr,
         public_threshold, public_auc) = testApproxSSLogReg(X_pub, y, X_test_pub, y_test)
    else:
        (public_f1score, public_accuracy,
         public_fpr, public_tpr,
         public_threshold, public_auc) = testLogReg(X_pub, y, X_test_pub, y_test)

    return (public_f1score, public_accuracy,
            public_fpr, public_tpr,
            public_threshold, public_auc)


def get_aim_W_and_data(pgm_train_df, domain, target, marginals_pgm, epsilon, delta, model_size, max_iters, n_samples):
    pgm_dataset = Dataset(pgm_train_df, domain)
    mrgs = selectTargetMarginals(pgm_train_df.columns, target, mode=marginals_pgm)
    mrgs_wkld = Workload((mrg, sparse.identity) for mrg in mrgs)
    pgm_synth = PGMsynthesizer(pgm_dataset, epsilon, delta, mrgs_wkld, model_size, max_iters, n_samples)
    pgm_synth.aimsynthesizer()
    ans_wkld = pgm_synth.ans_wkld
    W = {key: ans_wkld[key].__dict__['values'] for key in ans_wkld}
    synth = pgm_synth.synth.df
    synth_X, synth_y = synth.loc[:, synth.columns != target], synth.loc[:, synth.columns == target]

    return W, synth_X, synth_y


def dp_query_approx_ss_method(W_expanded, attribute_dict, training_columns, encoded_features, target, domain, n_train,
                              cols_to_dummy, one_hot, X_test, y_test):

    all_attributes_expanded = training_columns.append(pd.Index([target]))
    ZTZ = get_ZTZ(W_expanded, attribute_dict, all_attributes_expanded, cols_to_dummy, rescale=True)
    XTX = ZTZ.loc[training_columns, training_columns]
    XTy = ZTZ.loc[training_columns, target]

    if one_hot:
        X_test = normalize_minus1_1(X_test, attribute_dict, encoded_features)

    (DPapprox_f1score, DPapprox_accuracy,
     DPapprox_fpr, DPapprox_tpr,
     DPapprox_threshold, DPapprox_auc) = testPrivApproxSSLogReg(XTy, XTXy2, X_test, y_test, n_train, C=1.0)

    return (DPapprox_f1score, DPapprox_accuracy, DPapprox_fpr, DPapprox_tpr, DPapprox_threshold, DPapprox_auc)


def dp_query_synth_data_method(synth_X, synth_y, training_columns, attribute_dict, cols_to_dummy, encoded_features, X_test, y_test, one_hot):
    # Here we handle the case in which we have to one-hot encode the columns
    if one_hot:
        # As above we drop the first column in Pandas to avoid multi-collinearity
        # issues in the train data. We then check whether there is a column with all 0 or all 1
        # and we remove it. The trick by which we remove the columns with the same value is to see
        # which columns have a standard deviation equal to 0 or not
        # Here we set drop_first = False because we don't know which level for the synthetic AIM data would
        # be the first. Since we are enforcing the columns to be the same as the training data above,
        # the filtering at the column level later should get rid of collinearity issues.
        synth_X_ohe = pd.get_dummies(synth_X, columns=cols_to_dummy, drop_first=False)
        synth_X_ohe.drop(synth_X_ohe.std()[synth_X_ohe.std() == 0].index, axis=1, inplace=True)

        # Now we only select the columns that are present in the train set
        synth_X_ohe = synth_X_ohe[[el for el in synth_X_ohe.columns if el in training_columns]]
        aim_columns = synth_X_ohe.columns

        # Now we also modify the X_test so that we are sure that X_test also has the same columns as
        # the synthetic version of the data generated by AIM.
        X_test_aim = add_and_subsets_cols_to_test_set(X_test, aim_columns)
        synth_X = synth_X_ohe.copy()
    else:
        X_test_aim = X_test

    synth_X_ordered = pd.DataFrame()

    for col in X_test_aim.columns:
        synth_X_ordered[col] = synth_X[col]

    if one_hot:
        synth_X_ordered = normalize_minus1_1(synth_X_ordered, attribute_dict, encoded_features)
        X_test_aim = normalize_minus1_1(X_test_aim, attribute_dict, encoded_features)

    (aimsynth_f1score, aimsynth_accuracy,
     aimsynth_fpr, aimsynth_tpr,
     aimsynth_threshold, aimsynth_auc) = testLogReg(synth_X_ordered, synth_y, X_test_aim, y_test)

    return (aimsynth_f1score, aimsynth_accuracy, aimsynth_fpr,
            aimsynth_tpr, aimsynth_threshold, aimsynth_auc)


def objective_perturbation_method(X, y, X_test, y_test, attribute_dict, target, cols_to_dummy, epsilon, delta, one_hot):

    n, d = X.shape
    # maxvals = np.array(np.amax(X.to_numpy(), axis = 0)).reshape(-1,1)
    # max_row_norm = np.sqrt(np.sum([a**2 for a in maxvals]))
    if one_hot:
        encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
        X_gop = normalize_minus1_1(X, attribute_dict, encoded_features)
        X_test_gop = normalize_minus1_1(X_test, attribute_dict, encoded_features)
        bound_y = np.abs(max(attribute_dict[target]))
        max_row_norm = np.sqrt(len(attribute_dict.keys())-1)   # excludes target
        print(f"bound_y {bound_y}")
        print(f"max_row_norm {max_row_norm}")
    else:
        X_gop = X
        X_test_gop = X_test
        _, max_row_norm = get_bound_XTX(attribute_dict, target, cols_to_dummy, one_hot, rescale=False)

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
    Delta, b = genobjpert_get_params(X_gop.to_numpy(), epsilon, delta, lmda, zeta)

    # Run the iteration
    # fmin_tnc returns the convergence message as third argument in its output
    # 0 means local minimium reached, 1 and 2 means convergence by function value or theta value
    # 3 is maximum number of iterations reached, 4 linear search failed
    finish_opt = False
    patience = 3

    while not finish_opt and patience > 0:

        theta0 = np.random.normal(loc=0, scale=0.01, size=X.shape[1]).reshape(-1, )
        theta_opt = fmin_tnc(func=dp_objective, x0=theta0, fprime=dp_gradient, maxfun=10000, disp=0,
                             args=(X_gop.to_numpy(), y.to_numpy().reshape(-1, ), n, d, Delta, b))

        theta_final, n_it_run, final_message = theta_opt

        if final_message in [0, 1, 2]:
            finish_opt = True
        else:
            patience -= 1

    logits = np.dot(X_test_gop, theta_final)
    probabilities = 1 / (1 + np.exp(-logits))
    genobj_auc = roc_auc_score(y_test, probabilities)

    return genobj_auc