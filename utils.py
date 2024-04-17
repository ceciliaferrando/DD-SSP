import math
import numpy as np
from scipy import sparse
import pandas as pd
import copy
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
import pickle
import dill

import sys
sys.path.append('..')

# from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
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
from private_pgm_local.src.mbi.workload import Workload



def preprocess_data(dataset, target_dict, n_limit, one_hot, scale_y):
    """
    Preprocesses the data for further analysis.

    Parameters:
        dataset (str): Name of the dataset.
        target_dict (dict): Dictionary containing target information for datasets.
        n_limit (int): Limit on the number of data points.
        train_ratio (float): Ratio of training data.
        one_hot (bool): Whether to perform one-hot encoding.

    Returns:
        X (df): train features
        X_test (df): test features
        y (df): train target
        y_test (df): test target
        pgm_train_df (df): data for training AIM (AIM processes raw categorical tabular data, pre encoding)
        domain (dict): attribute domain
        target (str): name of the chosen predicted variable
        attribute_dict (dict): attrinbute information in the form {'attr_name': [list of possible values]}
        features_to_encode (list): columns to encode
        encoded_features (list): list of feature names post encoding
        original_ranges (dict): dictionary of the attribute ranges in the form {'attr_name': [min, max]}
        all_columns (list): list of the features included in training phase
    """

    # Import the data
    csv_path = '../hd-datasets-master/clean/' + dataset + '.csv'
    meta_path = '../hd-datasets-master/clean/' + dataset + '-domain.json'
    data = Dataset.load(csv_path, meta_path)  # for PGM
    domain = data.domain
    target = target_dict[dataset]

    df_train_path = f"{dataset}_train.csv"
    df_test_path = f"{dataset}_test.csv"

    if os.path.exists(df_train_path):
        train_df = pd.read_csv(df_train_path)
        test_df = pd.read_csv(df_test_path)

    else:
        df = pd.read_csv(csv_path)
        train_df, test_df = splitdf(df)
        train_df.to_csv(df_train_path, index = False)
        test_df.to_csv(df_test_path, index = False)

    train_df_n = train_df[:n_limit]

    X, y = train_df_n.loc[:, train_df_n.columns != target], train_df_n.loc[:, train_df.columns == target]
    X_test, y_test = test_df.loc[:, test_df.columns != target], test_df.loc[:, test_df.columns == target]

    pgm_train_df = X.join(y)
    for attr in domain:
        if attr not in pgm_train_df.columns:
            pgm_train_df[attr] = 0

    def Union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    all_original_columns = Union(train_df.columns, test_df.columns)

    # Create dictionary with attribute levels
    attribute_dict = {}

    for col in all_original_columns:
        unique_values = list(range(domain[col]))
        attribute_dict[col] = unique_values

    # If one_hot is active, then we one hot both the train set and the test set.
    features_to_encode, all_columns = [], X.columns

    if one_hot:
        X_pre = X
        X_test_pre = X_test
        print(f"one-hot encoding {dataset}...")
        features_to_encode = get_features_to_encode(dataset)
        X_ohe = one_hot_encode(X, features_to_encode, attribute_dict)
        X_test_ohe = one_hot_encode(X_test, features_to_encode, attribute_dict)
        assert set(X_ohe.columns) == set(X_test_ohe.columns)
        X = X_ohe.copy(deep=True)
        X_test = X_test_ohe.copy()

    all_columns = pd.Index(Union(X.columns, X_test.columns))
    encoded_features = [col for col in X if col.split("_")[0] in features_to_encode]
    original_ranges = {feature: [0, domain[feature]] for feature in attribute_dict.keys()}

    if one_hot:
        X = normalize_minus1_1(X, attribute_dict, encoded_features)
        X_test = normalize_minus1_1(X_test, attribute_dict, encoded_features)
        if scale_y:
            y = normalize_minus1_1(y, attribute_dict, encoded_features)
            y_test = normalize_minus1_1(y_test, attribute_dict, encoded_features)

    zero_std_cols = []
    for col in X.columns:
        if np.std(X[col]) == 0:
            print(f"feature {col} is a zero vector! Dropping it from training dataset")
            zero_std_cols.append(col)
    X.drop(columns=zero_std_cols, inplace=True)

    X_test = X_test[all_columns]

    return (X, X_test, y, y_test, X_pre, X_test_pre, pgm_train_df, domain, target, attribute_dict,
            features_to_encode, encoded_features, original_ranges, all_columns, zero_std_cols)


def splitdf(df):
    n = len(df)

    idxs = np.array(range(n))
    np.random.shuffle(idxs)

    n_test = 1000

    train_rows, test_rows = idxs[:(n-n_test)], idxs[(n-n_test):]
    df_train, df_test = df.iloc[train_rows, :], df.iloc[test_rows, :]

    return (df_train, df_test)


def get_features_to_encode(dataset):
    if dataset == "adult":
        features_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                              'native-country']
    elif dataset == "fire":
        features_to_encode = ['ALS Unit', 'Battalion', 'Call Final Disposition', 'Call Type', 'Call Type Group', 'City',
                              'Fire Prevention District', 'Neighborhooods - Analysis Boundaries', 'Station Area',
                              'Supervisor District', 'Unit Type', 'Zipcode of Incident']
    elif dataset == "taxi":
        features_to_encode = ['RatecodeID', 'PULocationID', 'DOLocationID', 'paymenttype']
    elif dataset == " stroke":
        features_to_encode = ['CMPLASP', 'CMPLHEP', 'CNTRYNUM', 'COUNTRY', 'DALIVE', 'DAP', 'DASP14', 'DASPLT',
                              "DAYLOCAL", "DCAA", "DCAREND", "DDEAD", "DDEADC", "DDEADD", "DDEADX",  "DDIAGHA", "DDIAGISC",
                              "DDIAGUN", "DGORM", "DHAEMD",
                              "DHH14", "DIED", "DIVH", "DLH14", "DMAJNCH", "DMAJNCHD", "DMAJNCHX", "DMH14", "DNOSTRK",
                              "DNOSTRKX", "DOAC", "DPE", "DPED", "DPLACE", "DRSH", "DRSHD", "DRSISC", "DRSISCD", "DRSUNK",
                              "DRSUNKD", "DSCH", "DSIDE", "DSIDED", "DSIDEX", "DSTER", "DTHROMB", "DVT14"]
    elif dataset == "ACSincome" or dataset == "ACSincome-LIN":
        features_to_encode = ['COW', 'MAR', 'RELP', 'RAC1P']
    elif dataset == "ACSemployment":
        features_to_encode = ['MAR', 'RELP', 'CIT', 'MIL', 'ANC', 'RAC1P']
    elif dataset == "ACSmobility":
        features_to_encode = ['MAR', 'CIT', 'MIL', 'ANC', 'RELP', 'RAC1P', 'GCL', 'COW', 'ESR']
    elif dataset == "ACSPublicCoverage":
        features_to_encode = ['MAR', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'ESR', 'FER', 'RAC1P']
    elif dataset == "ACSTravelTime":
        features_to_encode = ['MAR', 'ESP', 'MIG', 'RELP', 'RAC1P', 'CIT', 'OCCP', 'JWTR']
    elif dataset == "titanic":
        features_to_encode = ['Pclass', 'Cabin', 'Embarked']
    return features_to_encode


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
    elif mode == "all-pairs-weighted":
        out = list(combinations(cols, 2))
        out = balance_target_pairs(out, target)
    elif mode == 'all-triplets':
        out = list(combinations(cols, 3))
    elif mode == "no-target-pairs":
        cols_new = list(cols)
        cols_new.remove(target)
        out = combinations(cols_new, 2)
    return out

def balance_target_pairs(out, target):
    target_pairs = [tup for tup in out if target in tup]
    total_len = len(out)
    y_marg_len = len(target_pairs)
    p = 1/2

    previous_diff = float('inf')
    best_b = 0

    for b in range(0, total_len):  # We start from 1 since b can't be 0
        current_diff = abs((y_marg_len + b * y_marg_len) - p * total_len)  # we want 50% target marginals
        print("b", b, "current diff", current_diff)
        if current_diff < previous_diff:
            best_b = b
            previous_diff = current_diff
        else:
            break  # If the difference starts increasing, we stop iterating
    to_extend = target_pairs * best_b
    out.extend(to_extend)

    return out







def get_bound_XTX(attribute_dict, target, one_hot):
    if not one_hot:  # then data is binary synthetic data
        bound_X = np.sqrt(np.sum([max(attribute_dict[f]) ** 2 for f in attribute_dict if f != target]))
        bound_XTX = bound_X ** 2

    elif one_hot:  # follow sensitivity computation as described in paper
        bound_XTX = len(attribute_dict.keys()) - 1  # excludes target
        bound_X = np.sqrt(bound_XTX)

    return bound_XTX, bound_X


def normalize_minus1_1(X, attribute_dict, encoded_features):
    original_ranges = {attr: [min(attribute_dict[attr]), max(attribute_dict[attr])]
                       for attr in attribute_dict.keys()}

    X_out = pd.DataFrame()

    for col in X.columns:

        # this is in case the test set has columns of zeros
        if len(set(X[col])) == 1:
            X_out[col] = X[col]

        # if the column corresponds to a categorical feature that has been one-hot encoded, keep it in domain [0, 1]
        if col in encoded_features:
            X_out[col] = X[col]

        # for all other features, rescale to [-1, 1]
        else:
            print(original_ranges)
            print(col)
            colmin = original_ranges[col][0]
            colmax = original_ranges[col][1]

            col_1s = (1 - (-1)) * ((X[col] - colmin) / (colmax - colmin)) - 1
            X_out[col] = col_1s

    return X_out

def inverse_normalize(scaled_v, original_min, original_max):
    original_range = original_max - original_min
    unscaled = [int((x + 1) * (original_range / 2) + original_min) for x in scaled_v]
    return unscaled

# AIM Utils ############################################################################################################


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


class Chebyshev:
    """
    Chebyshev(a, b, n, func)
    Given a function func, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.
    Method eval(x) yields the approximated function value.
    """

    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                             for k in range(n)]) for j in range(n)]

    def eval(self, x):
        a, b = self.a, self.b
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)  # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:  # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]  # Last step is different


def phi_logit(x):
    return -math.log(1 + math.exp(-x))


def logit_2(x):
    return math.log(1 + math.exp(x))


def get_ZTZ(W, attribute_dict, columns, features_to_encode, target, rescale, scale_y):
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
                if attr_a_orig not in features_to_encode and attr_a_orig != target:
                    a_range_min, a_range_max = min(a_values), max(a_values)
                    a_values = [(1 - (-1)) * ((val - a_range_min) / (a_range_max - a_range_min)) - 1 for val in
                                a_values]
                if attr_b_orig not in features_to_encode and attr_b_orig != target:
                    b_range_min, b_range_max = min(b_values), max(b_values)
                    b_values = [(1 - (-1)) * ((val - b_range_min) / (b_range_max - b_range_min)) - 1 for val in
                                b_values]
                if attr_a_orig == target and scale_y == True:
                    a_range_min, a_range_max = min(a_values), max(a_values)
                    a_values = [(1 - (-1)) * ((val - a_range_min) / (a_range_max - a_range_min)) - 1 for val in
                                a_values]
                if attr_b_orig == target and scale_y == True:
                    b_range_min, b_range_max = min(b_values), max(b_values)
                    b_values = [(1 - (-1)) * ((val - b_range_min) / (b_range_max - b_range_min)) - 1 for val in
                                b_values]

            # case 1: a and b are both ordinal
            if attr_a_orig not in features_to_encode and attr_b_orig not in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                for j, j_value in enumerate(a_values):
                    for k, k_value in enumerate(b_values):
                        ZTZ[attr_a][attr_b] += j_value * k_value * mu_ab[j, k]

            # case 2.1: a is ordinal, b is encoded
            elif attr_a_orig not in features_to_encode and attr_b_orig in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                t = int(float(
                    attr_b.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                ZTZ[attr_a][attr_b] = np.sum(np.multiply(mu_ab[:, t], a_values))

            # case 2.2: a is encoded, b is ordinal
            elif attr_a_orig in features_to_encode and attr_b_orig not in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                s = int(float(
                    attr_a.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                ZTZ[attr_a][attr_b] = np.sum(np.multiply(mu_ab[s, :], b_values))

            # case 3: a and b are both encoded
            elif attr_a_orig in features_to_encode and attr_b_orig in features_to_encode:
                s = int(float(
                    attr_a.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                t = int(float(
                    attr_b.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                ZTZ[attr_a][attr_b] = mu_ab[s, t]

    # copy lower tri to upper tri
    ZTZ = ZTZ + ZTZ.T - np.diag(np.diag(ZTZ))

    return ZTZ


def one_hot_encode(df, features_to_encode, attribute_dict):
    encoded_df = df.copy()

    for feature in features_to_encode:
        unique_levels = attribute_dict[feature]
        for level in unique_levels:
            encoded_df[f'{feature}_{level}'] = (df[feature] == level).astype(int)

    encoded_df.drop(features_to_encode, axis=1, inplace=True)

    # drop first
    for col in encoded_df.columns:
        if col.endswith("_0"):
            encoded_df.drop(col, axis=1, inplace=True)

    encoded_df = encoded_df.copy()

    return encoded_df

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

def get_dp_approx_ll(theta, yTX, XTXy2, a, b, c, n):
    dp_approx_ll = n * a + np.dot(theta, yTX) * b + np.dot(np.dot(theta, XTXy2), theta) * c
    return dp_approx_ll

class SSApproxLL():

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

        if self.penalty == None:
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
        return (y_pred_proba)

    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.theta.T)
        y_pred_proba = expit(z)
        y_pred = 2 * ((y_pred_proba >= threshold).astype(int)) - 1
        print("y pred dpapproxss", y_pred)
        return (y_pred)

def get_aim_model(pgm_train_df, domain, target, marginals_pgm, epsilon, delta, model_size, max_iters, n_samples, initial_cliques):
    pgm_dataset = Dataset(pgm_train_df, domain)
    mrgs = selectTargetMarginals(pgm_train_df.columns, target, mode=marginals_pgm)
    mrgs = {(m, 1.0) for m in mrgs}
    mrgs_wkld = Workload(mrgs)
    y_pairs = [cl[0] for cl in mrgs_wkld if target in cl[0]] if initial_cliques == "y_marginals" else None
    print(f"initial cliques set to {y_pairs}")
    aim_model = aim.AIM(epsilon=epsilon, delta=delta, max_iters=max_iters, max_model_size=model_size)
    aim_model, synth  = aim_model.run(pgm_dataset, mrgs_wkld, n_samples, initial_cliques=y_pairs)
    return aim_model, mrgs_wkld


def get_unnormalized_prob(x, potentials):
    #print("potentials", potentials)
    sum_potentials = 0
    for factor in potentials:
        #print("factor", factor)
        #print("potentials[factor]", potentials[factor])
        index = tuple([int(x[attr]) for attr in factor])
        #print("index", index)
        factor_value = potentials[factor].values[index]
        #print("factor_value", factor_value)
        sum_potentials += factor_value

    return sum_potentials


def pred_y_given_x_from_G(x, G, target, target_levels):
    potentials = G.potentials
    unnormalized_probs = {}

    # print("potentials", potentials)
    # print("target_levels", target_levels)

    for level in range(len(target_levels)):
        # print("level", level)
        x_and_y = x.copy()
        x_and_y[target] = level
        unnormalized_probs[level] = get_unnormalized_prob(x_and_y, potentials)
        # print("unnormalized_probs[level]", unnormalized_probs[level])

    list_guesses = list(unnormalized_probs.items())
    list_guesses.sort(key=lambda x: x[1])
    best_guess = list_guesses[-1][0]

    return best_guess

def testLogReg(theta, X_test, y_test):
    logits = np.dot(X_test, theta)
    probabilities = 1 / (1 + np.exp(-logits))
    auc = roc_auc_score(y_test, probabilities)
    return auc

def testLinReg(theta, X_test, y_test):
    y_pred = np.dot(X_test, theta)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def public_logreg(X, y, all_columns):
    model = LogisticRegression(penalty=None, fit_intercept=False, max_iter=2000)
    model.fit(X.to_numpy(), y.to_numpy().ravel())
    theta = model.coef_.ravel()   # probabilities for class 1
    theta = pd.DataFrame(theta, index=X.columns)
    theta = theta.reindex(index=all_columns, fill_value=0)

    return theta

def dp_query_approx_ss_logreg(ZTZ, all_columns, target, n, cheb, C=1.0):

    XTXy2 = ZTZ.loc[all_columns, all_columns]
    XTy = ZTZ.loc[all_columns, target]
    alpha = 1 / (n * C)
    model = SSApproxLL(cheb, XTy, XTXy2, n, penalty=None, alpha=alpha)
    model.fit()
    theta = model.theta
    theta = pd.DataFrame(theta)
    theta.set_index(all_columns, inplace=True)

    return theta

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

def public_linreg(X, y):

    n, d = X.shape

    X_vec, y_vec = X.values, y.values.ravel()
    XTX = np.dot(X_vec.T, X_vec)
    XTX = XTX + np.eye(d) * 1e-12
    XTy = np.dot(X_vec.T, y_vec)
    theta = np.linalg.solve(XTX, XTy)

    # regr = LinearRegression(fit_intercept=False)
    # regr.fit(X.values, y.values.ravel())
    # theta = regr.coef_
    theta = pd.DataFrame(theta)
    theta = theta.set_index(X.columns)

    return theta

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

def dp_query_ss_linreg(ZTZ, all_columns, target):

    d = len(all_columns)

    XTX = ZTZ.loc[all_columns, all_columns]
    XTy = ZTZ.loc[all_columns, target]

    # get estimator
    XTX = XTX + np.eye(d) * 1e-12
    theta_query_ss = np.linalg.solve(XTX, XTy)
    theta_query_ss = pd.DataFrame(theta_query_ss)
    theta_query_ss = theta_query_ss.set_index(all_columns)

    return theta_query_ss, XTX, XTy


def get_SS_mse(XTX_private, XTX_public, XTy_private, XTy_public):
    if not XTX_private.index.equals(XTX_public.index):
        XTX_private, XTX_public = XTX_private.align(XTX_public, axis=0, join='outer', fill_value=0)
        XTX_private, XTX_public = XTX_private.align(XTX_public, axis=1, join='outer', fill_value=0)

    if not XTy_private.index.equals(XTy_public.index):
        XTy_private, XTy_public = XTy_private.align(XTy_public, axis=0, join='outer', fill_value=0)

    return (np.sqrt(mean_squared_error(XTX_private, XTX_public)),
            np.sqrt(mean_squared_error(XTy_private, XTy_public)))


