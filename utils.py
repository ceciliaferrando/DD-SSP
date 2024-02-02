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




########## GENERAL UTILS ################################################################


def splitdf(df, target, train_ratio):
    n = len(df)

    idxs = np.array(range(n))
    np.random.shuffle(idxs)

    train_rows, test_rows = idxs[:int(train_ratio * n)], idxs[int(train_ratio * n):]
    df_train, df_test = df.iloc[train_rows, :], df.iloc[test_rows, :]

    df_X_train, df_y_train = df_train.loc[:, df_train.columns != target], df_train.loc[:, df_train.columns == target]
    df_X_test, df_y_test = df_test.loc[:, df_test.columns != target], df_test.loc[:, df_test.columns == target]

    return (df_X_train, df_y_train, df_X_test, df_y_test)


def get_cols_to_dummy(dataset):
    if dataset == "adult":
        cols_to_dummy = ['workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'race', 'native-country']
    elif dataset == "ACSincome":
        cols_to_dummy = ['COW', 'MAR', 'RELP', 'RAC1P']
    elif dataset == "ACSemployment":
        cols_to_dummy = ['MAR', 'RELP', 'CIT', 'MIL', 'ANC',
                         'RAC1P']
    elif dataset == "ACSmobility":
        cols_to_dummy = ['MAR', 'CIT', 'MIL', 'ANC', 'RELP',
                         'RAC1P', 'GCL', 'COW', 'ESR']
    elif dataset == "ACSPublicCoverage":
        cols_to_dummy = ['MAR', 'ESP', 'CIT', 'MIG', 'MIL',
                         'ANC', 'ESR', 'FER', 'RAC1P']
    elif dataset == "ACSTravelTime":
        cols_to_dummy = ['MAR', 'ESP', 'MIG', 'RELP', 'RAC1P', 'CIT',
                         'OCCP', 'JWTR']
    elif dataset == "titanic":
        cols_to_dummy = ['Pclass', 'Cabin', 'Embarked']
    elif dataset == "diabetes":
        cols_to_dummy = None
    return cols_to_dummy


def add_and_subsets_cols_to_test_set(X_test, columns):
    for col_val in columns:
        if col_val not in X_test.columns:
            X_test[col_val] = 0

    return X_test[columns]


def selectTargetMarginals(cols, target, mode='target-pairs'):
    out = []
    if mode == 'target-pairs':
        for col in cols:
            if col != target:
                out.append((col, target))
    elif mode == 'target-triplets':
        cols_new = list(cols)
        cols_new.remove(target)
        tmp_pairs = combinations(cols_new, 2)
        out = [(t[0], t[1], target) for t in tmp_pairs]
    elif mode == 'target-pairs-target-triplets':
        out = []
        for col in cols:
            if col != target:
                out.append((col, target))
        cols_new = list(cols)
        cols_new.remove(target)
        tmp_pairs = combinations(cols_new, 2)
        out.extend([(t[0], t[1], target) for t in tmp_pairs])
    elif mode == "all-pairs":
        out = list(combinations(cols, 2))
    elif mode == 'all-triplets':
        out = list(combinations(cols, 3))
    elif mode == "no-target-pairs":
        cols_new = list(cols)
        cols_new.remove(target)
        out = combinations(cols_new, 2)
    return out


def get_bound_XTX(attribute_dict, target, features_to_encode, one_hot, rescale):

    if not one_hot: # then data is binary synthetic data
        bound_X = np.sqrt(np.sum([max(attribute_dict[f])**2 for f in attribute_dict if f!=target]))
        bound_XTX = bound_X**2
    elif one_hot and rescale:
        bound_XTX = len(attribute_dict.keys()) - 1    #excludes target
        bound_X = np.sqrt(bound_XTX)
    elif one_hot and not rescale:
        print("this option is not covered")

    return bound_XTX, bound_X

def testLogReg(synth_X, synth_y, X_test, y_test):
    lr = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000, warm_start=True)
    lr.fit(synth_X, synth_y)
    pred_y, prob_y_1 = lr.predict(X_test), lr.predict_proba(X_test)[:, 1]
    f1score = f1_score(y_test, pred_y)
    accuracy = accuracy_score(y_test, pred_y)
    fpr, tpr, threshold = roc_curve(y_test, prob_y_1)  # check that it's the actual values not the thresholded
    roc_auc = auc(fpr, tpr)
    return (f1score, accuracy, fpr, tpr, threshold, roc_auc)


def testLogRegCustom(synth_X, synth_y, X_test, y_test, InvC):
    lr = LogisticRegression(penalty="l2", C=InvC, fit_intercept=False, max_iter=1000, warm_start=True)
    lr.fit(synth_X, synth_y)
    pred_y, prob_y_1 = lr.predict(X_test), lr.predict_proba(X_test)[:, 1]
    f1score = f1_score(y_test, pred_y)
    accuracy = accuracy_score(y_test, pred_y)
    fpr, tpr, threshold = roc_curve(y_test, prob_y_1)  # check that it's the actual values not the thresholded
    roc_auc = auc(fpr, tpr)
    return (f1score, accuracy, fpr, tpr, threshold, roc_auc)


def normalize_minus1_1(X, encoded_features, original_X_range):
    X_out = pd.DataFrame()

    for col in X.columns:
        # this is in case the test set has columns of zeros
        if len(set(X[col])) == 1:
            X_out[col] = X[col]

        # if the column corresponds to a categorical feature that has beenn one-hot encoded, keep it in domain [0, 1]
        elif col in encoded_features:
            X_out[col] = X[col]

        # for all other features, rescale to [-1, 1]
        else:
            colmin = original_X_range[col][0]
            colmax = original_X_range[col][1]

            col_1s = (1 - (-1)) * ((X[col] - colmin) / (colmax - colmin)) - 1
            X_out[col] = col_1s

    return X_out

def theta_inverse_mapping(theta, original_ranges, training_columns):

    theta_mapped_back = np.zeros((training_columns,))

    for j, col in enumerate(training_columns):
        colmin = np.min(original_ranges[col])
        colmax = np.max(original_ranges[col])
        theta_j_mapped_back = (theta[j] + 1)/2 * (colmax-colmin) + colmin
        theta_mapped_back[j] = theta_j_mapped_back

    return theta_mapped_back

# AIM SYNTHESIZER

class PGMsynthesizer():

    def __init__(self, data, epsilon, delta, measurements, model_size, max_iters, num_synth_rows):
        self.epsilon = epsilon
        self.delta = delta
        self.data = data
        self.measurements = measurements
        self.model_size = model_size
        self.max_iters = max_iters
        self.num_synth_rows = num_synth_rows
        self.synth = None
        self.G = None

    def mstsynthesizer(self):
        self.synth, self.G = mst.mst(self.data, self.epsilon, self.delta)

    def mwemsynthesizer(self):
        self.synth, self.G = mwem.mwem_pgm(self.data, self.epsilon, workload=self.measurements)

    def v13synthesizer(self):
        v13model = v13.V13(self.epsilon, self.delta)
        self.synth, self.G = v13model.run(self.data, self.measurements)

    def aimsynthesizer(self):
        aimmodel = aim.AIM(epsilon=self.epsilon, delta=self.delta, max_iters=self.max_iters,
                           max_model_size=self.model_size)
        self.synth, self.G, self.ans_wkld = aimmodel.run(self.data, self.measurements, self.num_synth_rows,
                                                         output_graphical_model=True)


# DQ Query Approximation Utils #########################################################################################


def expand_W(W, attribute_dict):
    # add symmetric entries
    W_expanded = copy.deepcopy(W)
    for el in W:
        W_expanded[el[1], el[0]] = W[el].T

    # add (x, x) pairs
    for col in attribute_dict.keys():
        table_with_col = [W_expanded[tple] for tple in W_expanded if tple[0] == col][0]
        col_counts = np.sum(table_with_col, axis=1)
        W_expanded[col, col] = np.diag(col_counts)

    return W_expanded


def get_ZTZ(W, attribute_dict, columns, features_to_encode, rescale):
    """
        W: [dict] marginal tables in the form {("feature_A", "feature_B"); m_A x m_B np.array of counts, ...}
        attribute_dict: [dict] attribute levels, {"attr_A": list of ordered possible levels for attr_A, ...}
                                - should include target
        columns: [list] list of names of post-encoding attributes
        features_to_encode: [list] list of features requiring 1-hot encoding}
        rescale: [boolean] True if rescaling numerical non binary attributes in [-1, 1], TBC
    """

    # initialize ZTZ as a DataFrame with *named* columns and rows
    base_matrix = np.zeros((len(columns), len(columns)))
    ZTZ = pd.DataFrame(base_matrix, columns=columns, index=columns)

    # loop through attribute pairs

    for a, attr_a in enumerate(columns):
        for b, attr_b in enumerate(columns[a:]):

            # root name of the attributes
            attr_a_orig = attr_a.split("_")[0]
            attr_b_orig = attr_b.split("_")[0]
            # possible level values
            a_values = attribute_dict[attr_a_orig]
            b_values = attribute_dict[attr_b_orig]
            if rescale:
                if attr_a_orig not in features_to_encode:
                    a_range_min, a_range_max = min(a_values), max(a_values)
                    a_values = [(1 -(-1)) * ((val - a_range_min) / (a_range_max - a_range_min)) - 1 for val in a_values]
                if attr_b_orig not in features_to_encode:
                    b_range_min, b_range_max = min(b_values), max(b_values)
                    b_values = [(1 -(-1)) * ((val - b_range_min) / (b_range_max - b_range_min)) - 1 for val in b_values]

            # case 1: a and b are both ordinal
            if attr_a_orig not in features_to_encode and attr_b_orig not in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                for j, j_value in enumerate(a_values):
                    for k, k_value in enumerate(b_values):
                        ZTZ[attr_a][attr_b] += j_value * k_value * mu_ab[j, k]

            # case 2.1: a is ordinal, b is encoded
            elif attr_a_orig not in features_to_encode and attr_b_orig in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                t = int(attr_b.split("_")[-1])  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                ZTZ[attr_a][attr_b] = np.sum(np.multiply(mu_ab[:, t], a_values))

            # case 2.2: a is encoded, b is ordinal
            elif attr_a_orig in features_to_encode and attr_b_orig not in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                s = int(attr_a.split("_")[-1])  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                ZTZ[attr_a][attr_b] = np.sum(np.multiply(mu_ab[s, :], b_values))

            # case 3: a and b are both encoded
            elif attr_a_orig in features_to_encode and attr_b_orig in features_to_encode:
                s = int(attr_a.split("_")[-1])  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                t = int(attr_b.split("_")[-1])  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                ZTZ[attr_a][attr_b] = mu_ab[s, t]

    # copy lower tri to upper tri
    ZTZ = ZTZ + ZTZ.T - np.diag(np.diag(ZTZ))

    return ZTZ