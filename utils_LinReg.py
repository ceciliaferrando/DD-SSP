import numpy as np
import pandas as pd
import json
import copy
from scipy import sparse

import sys
sys.path.append('..')

from mbi import Dataset
from dpsynth.workload import Workload
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from private_pgm_local.mechanisms import aim

from itertools import combinations
import pickle
import pdb

import os

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import argparse
from tqdm.auto import tqdm


def public_linear_regression(X, y, X_test, y_test):
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    theta_public = np.linalg.solve(XTX, XTy)
    # reg = LinearRegression(fit_intercept=False).fit(X, y)
    # theta_public = reg.coef_.T

    y_pred = np.dot(X_test, theta_public)
    mse_score_public = mean_squared_error(y_test, y_pred)
    r2_score_public = r2_score(y_test, y_pred)

    return theta_public, mse_score_public, r2_score_public


def AdaSSP_linear_regression(X, y, epsilon, delta, rho, bound_X, bound_y, bound_XTX, X_test, y_test, original_y_range,
                             rescale):
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
    sensitivity = np.sqrt(np.log(6 / delta)) / (epsilon / 3)
    eigen_min_dp = max(0,
                       eigen_min + sensitivity * (bound_XTX) * z -
                       (bound_XTX) * np.log(6 / delta) / (epsilon / 3))
    lambda_dp = max(0,
                    np.sqrt(d * np.log(6 / delta) * np.log(2 * (d ** 2) / rho)) * (bound_XTX) /
                    (epsilon / 3) - eigen_min_dp)

    tri = np.triu(np.random.normal(0, 1, (d, d)))
    Zsym = tri + tri.T - np.diag(np.diag(tri))
    XTX_dp = XTX + sensitivity * (bound_XTX) * Zsym
    print("epsilon =", epsilon, "XTX_dp noise variance", sensitivity * (bound_XTX))

    z = np.random.normal(0, 1, size=(d,))
    XTy_dp = XTy + sensitivity * bound_X * bound_y * z
    XTX_dp_reg = XTX_dp + lambda_dp * np.eye(d)

    theta_dp = np.linalg.solve(XTX_dp_reg, XTy_dp)

    y_pred = np.dot(X_test, theta_dp)

    # scale y pred back to original domain if rescale needed. Linear rescaling.
    if rescale:
        y_pred = (y_pred - (-1)) / (+1 - (-1)) * (original_y_range[1] - original_y_range[0]) + original_y_range[0]
        y_pred = y_pred.astype(int)
    mse_dp = mean_squared_error(y_test, y_pred)
    r2_dp = r2_score(y_test, y_pred)

    return theta_dp, mse_dp, r2_dp


def splitdf(df, target, train_ratio):
    """Split the dataframe into train and test

    Args:
        df: data in the form of Pandas dataframe
        target: name of the target variable (str)
        train_ratio: proportion of train data (e.g. 0.7 for 70%)

    Returns:
        df_X_train, df_y_train, df_X_test, df_y_test
    """

    n = len(df)

    idxs = np.array(range(n))
    np.random.shuffle(idxs)

    train_rows, test_rows = idxs[:int(train_ratio * n)], idxs[int(train_ratio * n):]
    df_train, df_test = df.iloc[train_rows, :], df.iloc[test_rows, :]

    df_X_train, df_y_train = df_train.loc[:, df_train.columns != target], df_train.loc[:, df_train.columns == target]
    df_X_test, df_y_test = df_test.loc[:, df_test.columns != target], df_test.loc[:, df_test.columns == target]

    return (df_X_train, df_y_train, df_X_test, df_y_test)


def get_cols_to_dummy(dataset):
    """Returns the list of categorical variables to be one-hot encoded

    Args:
        dataset: name of the dataset (str)

    Returns:
        cols_to_dummy: list of features that need to be one-hot encoded
    """

    if dataset == "adult":
        cols_to_dummy = ['workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'native-country']
    elif dataset in ["ACSincome", "ACSincome-LIN"]:
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
    """Returns the list of variable cliques representing the marginals of interest

    Args:
        cols: list of variable names
        target: name of the target variable (str)
        mode: type of marginals (str), default 'target-pairs'

    Returns:
        out: workload (list) collecting the marginals of interest
    """

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


def get_one_way_from_W(W, features):
    one_way = {}
    for feature in features:

        # feature = feature.split("_")[0]
        tuples_with_feature = [tuple(t) for t in W.keys() if feature in t]
        # select a random tuple from the list
        random_tuple = tuples_with_feature[0]
        if random_tuple.index(feature) == 0:
            feature_counts = np.sum(W[random_tuple], axis=1)
        elif random_tuple.index(feature) == 1:
            feature_counts = np.sum(W[random_tuple], axis=0)
        one_way[feature] = feature_counts

    return one_way


def rescale_value_minus1_1(val, original_range):
    colmin, colmax = original_range
    rescaled_val = (1 -(-1)) * ((val - colmin) / (colmax - colmin)) - 1
    return rescaled_val


def get_XTXjk(Cjk, j_values, k_values):
    XTXjk = 0
    for j, j_value in enumerate(j_values):
        for k, k_value in enumerate(k_values):
            XTXjk += j_value * k_value * Cjk[j, k]
    return XTXjk


def get_XTXjj(Cj, j_values):
    XTXjj = 0
    for j_index, j_value in enumerate(j_values):
        XTXjj += j_value * j_value * Cj[j_index]
    return XTXjj


def get_XTX_from_ans_wkld(ans_wkld, feature_dict, features, features_to_encode, rescale):
    one_way = get_one_way_from_W(ans_wkld, features)
    features = list(one_way.keys())
    # from all pairs
    d = len(features)
    XTX_dict = {}
    for j in range(d):
        for k in range(d):
            if j != k and (features[j], features[k]) in ans_wkld:
                pair = (features[j], features[k])
                Cjk = np.array(ans_wkld[pair])
                j_values = feature_dict[features[j]]
                k_values = feature_dict[features[k]]
                if features[j].split("_")[0] not in features_to_encode and rescale == True: # need to rescale in [-1, 1]
                    j_values = [rescale_value_minus1_1(val,
                                                       [min(feature_dict[features[j]]),
                                                        max(feature_dict[features[j]])]) for val in j_values]
                if features[k].split("_")[0]  not in features_to_encode and rescale == True: # need to rescale in [-1, 1]
                    k_values = [rescale_value_minus1_1(val, [min(feature_dict[features[k]]),
                                                        max(feature_dict[features[k]])]) for val in k_values]
                XTXjk = get_XTXjk(Cjk, j_values, k_values)
                XTX_dict[features[j], features[k]] = XTXjk
                XTX_dict[features[k], features[j]] = XTXjk
            elif j == k:
                j_values = feature_dict[features[j]]
                Cj = np.array(one_way[features[j]])
                XTXjk = get_XTXjj(Cj, j_values)
                XTX_dict[features[j], features[k]] = XTXjk

    return XTX_dict


def get_XTyj(Cj, j_values, feature_dict, target, target_values):
    XTyj = 0
    for k, kappa in enumerate(j_values):
        for y_idx, y_value in enumerate(target_values):
            XTyj += y_value * kappa * Cj[k, y_idx]
    return XTyj


def get_yTXj(Cj, j_values):
    yTXj = 0
    for k, kappa in enumerate(j_values):
        yTXj += +kappa * Cj[k, 1]
        yTXj += -kappa * Cj[k, 0]
    return yTXj

def get_XTy_from_ans_wkld(ans_wkld, feature_dict, features, features_to_encode, target, target_values, rescale):
    target_pairs = {key: ans_wkld[key] for key in ans_wkld if target in key and key[1 - key.index(target)] in features}
    features = list(feature_dict.keys())
    d = len(features)
    XTy_dict = {}
    for j, pair in enumerate(target_pairs):
        feature_name = pair[1 - pair.index(target)]
        if feature_name in features:
            if pair.index(target) == 0:
                Cj = copy.deepcopy(target_pairs[pair].T)
            else:
                Cj = copy.deepcopy(target_pairs[pair])
            j_values = feature_dict[feature_name]
            if feature_name.split("_")[0] not in features_to_encode and rescale == True:  # need to rescale in [-1, 1]
                j_values = [rescale_value_minus1_1(val,
                                                   [min(feature_dict[features[j]]),
                                                    max(feature_dict[features[j]])]) for val in j_values]
            if rescale:
                target_values = [rescale_value_minus1_1(val,
                                                   [min(target_values),
                                                    max(target_values)]) for val in target_values]
            XTyj = get_XTyj(Cj, j_values, feature_dict, target, target_values)
            XTy_dict[feature_name] = XTyj
    return XTy_dict


def get_XTXy2_from_ans_wkld(ans_wkld, one_way, features, feature_dict, target):
    # from all pairs
    features = [f for f in features if f != target]
    d = len(features)
    XTXy2 = np.zeros((d, d))
    for j in range(d):
        for k in range(d):
            if j != k and (features[j], features[k]) in ans_wkld:
                pair = (features[j], features[k])
                Cjk = np.array(ans_wkld[pair])
                j_values = feature_dict[features[j]]
                k_values = feature_dict[features[k]]
                XTXjk = get_XTXjk(Cjk, j_values, k_values)
                XTXy2[j, k] = XTXjk
                XTXy2[k, j] = XTXjk
            elif j == k:
                j_values = feature_dict[features[j]]
                Cj = np.array(one_way[features[j]])
                XTXjk = get_XTXjj(Cj, j_values)
                XTXy2[j, k] = XTXjk
    return XTXy2


def one_hot_encode_contingency(feature_dict, contingency_tables, variables_to_encode):
    """
        One-hot encodes AIM contingency tables
    """

    for var_to_encode in variables_to_encode:
        keys_to_check = list(contingency_tables.keys())
        for key in keys_to_check:
            table = contingency_tables[key]
            if key[1] == var_to_encode:
                levels = feature_dict[var_to_encode]

                encoded_matrix = np.zeros((len(table), len(levels), 2))

                for i in range(len(table)):
                    for k in range(len(levels)):
                        level = int(levels[k])
                        encoded_matrix[i, k, 1] = table[i, level]
                        encoded_matrix[i, k, 0] = sum(table[i]) - table[i, level]

                        contingency_tables[(key[0], var_to_encode + "_" + str(level))] = encoded_matrix[:, k, :]

                        keys_to_check.append((key[0], var_to_encode + "_" + str(level)))
                del contingency_tables[key]

    # repeat and check the first feature
    for var_to_encode in variables_to_encode:
        keys_to_check = list(contingency_tables.keys())
        for key in keys_to_check:
            table = contingency_tables[key]
            if key[0] == var_to_encode:
                levels = feature_dict[var_to_encode]

                encoded_matrix = np.zeros((len(levels), len(table[0]), 2))

                for i in range(len(levels)):
                    for k in range(len(table[0])):
                        level = int(levels[i])
                        encoded_matrix[i, k, 1] = table[level, k]
                        encoded_matrix[i, k, 0] = sum(table[:, k]) - table[level, k]

                        contingency_tables[(var_to_encode + "_" + str(level), key[1])] = encoded_matrix[i, :, :].T

                        keys_to_check.append((var_to_encode + "_" + str(level), key[1]))
                del contingency_tables[key]

    # drop first to avoid collinearity
    contingency_tables_final = {}
    for pair in contingency_tables:
        if not pair[0].endswith("_0") and not pair[1].endswith("_0"):
            contingency_tables_final[pair] = contingency_tables[pair]

    return contingency_tables_final


def add_and_subsets_cols_to_test_set(X_test, reference_columns):
    """
    - Add with all zeros the columns that are in the training data but not in the test data
    - Remove all other columns
    """

    for col_val in reference_columns:
        if col_val not in X_test.columns:
            X_test[col_val] = 0

    return X_test[reference_columns]


def get_bound_X(one_hot, feature_dict, features_to_encode, target):
    # differentiates between one-hot encoded data and raw data

    feature_dict = {k: feature_dict[k] for k in feature_dict if k != target}

    if one_hot:
        sum_squares = 0
        for feature in feature_dict:
            # if feature in features_to_encode:
            #     sum_squares += (1 - 0)**2
            # else:
            #     pox_values = feature_dict[feature]
            #     sum_squares += (max(pox_values) - min(pox_values)) ** 2
            pox_values = feature_dict[feature]
            if feature not in features_to_encode:
                sum_squares += max(np.abs(pox_values)) ** 2
        norm = np.sqrt(sum_squares) + len(features_to_encode)
        print("one-hot encoded aware norm", norm)

    else:
        sum_squares = 0
        for feature in feature_dict:
            pox_values = feature_dict[feature]
            # sum_squares += (max(pox_values) - min(pox_values)) ** 2
            sum_squares += max(np.abs(pox_values)) ** 2
        norm = np.sqrt(sum_squares)
    return norm


def get_bound_XTX_1h(feature_dict, features_to_encode, feature_dict_1h, target):
    features = [key for key in feature_dict_1h.keys() if key != target]
    seen = {}
    for i in range(len(features)):
        for j in range(i, len(features)):

            feature_1, feature_2 = features[i], features[j]
            pair = (feature_1, feature_2)

            if (feature_1, feature_2) not in seen and (feature_2, feature_1) not in seen:

                feature_1, feature_2 = feature_1.split("_")[0], feature_2.split("_")[0]

                R_1 = max(feature_dict[feature_1]) - min(feature_dict[feature_1])
                R_2 = max(feature_dict[feature_2]) - min(feature_dict[feature_2])

                # case 1: none of the two features and 1h encoded
                if feature_1 not in features_to_encode and feature_2 not in features_to_encode:
                    num_1_options = len(feature_dict[feature_1])
                    num_2_options = len(feature_dict[feature_2])
                    seen[pair] = (R_1 * R_2) ** 2
                # case 2: feature 1 is 1h encoded, feature 2 is not 1h encoded
                elif feature_1 in features_to_encode and feature_2 not in features_to_encode:
                    num_1_options = len(feature_dict[feature_1]) - 1
                    num_2_options = len(feature_dict[feature_2])
                    seen[pair] = ((1 * R_2) / num_1_options) ** 2
                # case 3: feature 2 is 1h encoded, feature 1 is not 1h encoded
                elif feature_2 in features_to_encode and feature_1 not in features_to_encode:
                    num_1_options = len(feature_dict[feature_1])
                    num_2_options = len(feature_dict[feature_2]) - 1
                    seen[pair] = ((R_1 * 1) / num_2_options) ** 2
                # case 4: both features are 1h encoded
                elif feature_1 in features_to_encode and feature_2 in features_to_encode:
                    num_1_options = len(feature_dict[feature_1]) - 1
                    num_2_options = len(feature_dict[feature_2]) - 1
                    seen[pair] = (1 / (num_1_options * num_2_options)) ** 2

    sum_of_squares = np.sum(list(seen.values()))
    upperbound_XTX_1h = np.sqrt(sum_of_squares)

    return (upperbound_XTX_1h)


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
            col_1s = 2 * ((X[col] - colmin) / (colmax - colmin)) - 1
            X_out[col] = col_1s

    return X_out


def data_exploration(dataset, df, ncols=5):
    num_cols = len(df.columns)
    nrows = (num_cols + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 3 * nrows))
    axes = axes.flatten()  # Flatten the 2D array of subplots

    for i, col in enumerate(df.columns):
        df[col].plot(kind='hist', ax=axes[i])
        axes[i].set_title(col + " distribution")

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()  # Improves appearance a bit.
    plt.show()
    fig.savefig(dataset + "_gridplot.png")