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

from private_pgm.src.mbi import Dataset, FactoredInference
from private_pgm.mechanisms import aim
from private_pgm.src.mbi.workload import Workload


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
    csv_path = 'hd-datasets-master/clean/' + dataset + '.csv'
    meta_path = 'hd-datasets-master/clean/' + dataset + '-domain.json'
    data = Dataset.load(csv_path, meta_path)  # for PGM
    domain = data.domain
    target = target_dict[dataset]

    df_train_path = f"data/{dataset}_train.csv"
    df_test_path = f"data/{dataset}_test.csv"

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
    """
    Input
        df: [pd dataframe] original data set as a pandas dataframe
    Returns
        df_train: [pd dataframe] train data set as a pandas dataframe
        df_test: [pd dataframe] test data set as a pandas dataframe
    """

    n = len(df)

    idxs = np.array(range(n))
    np.random.shuffle(idxs)

    n_test = 1000

    train_rows, test_rows = idxs[:(n-n_test)], idxs[(n-n_test):]
    df_train, df_test = df.iloc[train_rows, :], df.iloc[test_rows, :]

    return (df_train, df_test)


def get_features_to_encode(dataset):
    """
    Input
        dataset: [str] name of the dataset
    Returns
        features_to_encode: [list] list of features requiring one-hot encoding
    """

    if dataset == "adult":
        features_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                              'native-country']
    elif dataset == "fire":
        features_to_encode = ['ALS Unit', 'Battalion', 'Call Final Disposition', 'Call Type', 'Call Type Group', 'City',
                              'Fire Prevention District', 'Neighborhooods - Analysis Boundaries', 'Station Area',
                              'Supervisor District', 'Unit Type', 'Zipcode of Incident']
    elif dataset == "taxi":
        features_to_encode = ['RatecodeID', 'PULocationID', 'DOLocationID', 'paymenttype']
    elif dataset == "ACSincome" or dataset == "ACSincome-LIN":
        features_to_encode = ['COW', 'MAR', 'RELP', 'RAC1P']
    elif dataset == "ACSemployment":
        features_to_encode = ['MAR', 'RELP', 'CIT', 'MIL', 'ANC', 'RAC1P']
    elif dataset == "ACSmobility":
        features_to_encode = ['MAR', 'CIT', 'MIL', 'ANC', 'RELP', 'RAC1P', 'GCL', 'COW', 'ESR']
    elif dataset == "ACSPublicCoverage":
        features_to_encode = ['MAR', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'ESR', 'FER', 'RAC1P']
    return features_to_encode


def selectTargetMarginals(cols, target, mode):
    """
    Input
        cols: [list] list of column attributes
        target: [str] name of target variable
        mode: [str] type of marginals (e.g. "all-pairs")
    Returns
        out: [list] list of marginals
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
    """
    balances the number of target pairs in the output of selectTargetMarginal
    """
    target_pairs = [tup for tup in out if target in tup]
    total_len = len(out)
    y_marg_len = len(target_pairs)
    p = 1/2

    previous_diff = float('inf')
    best_b = 0

    for b in range(0, total_len):  # We start from 1 since b can't be 0
        current_diff = abs((y_marg_len + b * y_marg_len) - p * total_len)  # we want 50% target marginals
        if current_diff < previous_diff:
            best_b = b
            previous_diff = current_diff
        else:
            break  # If the difference starts increasing, we stop iterating
    to_extend = target_pairs * best_b
    out.extend(to_extend)

    return out


def get_bound_XTX(attribute_dict, target, one_hot):
    """
    Input
        attribute_dict [dict]: attribute information in the form {'attr_name': [list of possible values]}
        target: [str] name of target variable
        one_hot: [bool] whether one-hot encoding is used
    Returns
         bound_XTX, bound_X: [floats] upperbounds to the sensitivity of XTX and X respectively
    """

    if not one_hot:  # then data is binary synthetic data
        bound_X = np.sqrt(np.sum([max(attribute_dict[f]) ** 2 for f in attribute_dict if f != target]))
        bound_XTX = bound_X ** 2

    elif one_hot:
        bound_XTX = len(attribute_dict.keys()) - 1  # excludes target
        bound_X = np.sqrt(bound_XTX)

    return bound_XTX, bound_X


def normalize_minus1_1(X, attribute_dict, encoded_features):
    """
    Rescales numerical non-encoded features in range [-1,1]
    """

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
            colmin = original_ranges[col][0]
            colmax = original_ranges[col][1]

            col_1s = (1 - (-1)) * ((X[col] - colmin) / (colmax - colmin)) - 1
            X_out[col] = col_1s

    return X_out


# DDSSP Utils ##########################################################################################################

def expand_W(W, attribute_dict):
    """
    Input
        W: [dict] marginal tables in the form {("feature_A", "feature_B"); m_A x m_B np.array of counts, ...}
        attribute_dict: [dict] attribute levels, {"attr_A": list of ordered possible levels for attr_A, ...}
                                - should include target

    Returns
        W_expanded: [dict] W with added one-way marginals as diagonal matrices and symmetric two-way marginals
    """
    W_expanded = copy.deepcopy(W)

    # get symmetric tuples
    for el in W:
        W_expanded[el[1], el[0]] = W[el].T

    # add (x, x) pairs
    for col in attribute_dict.keys():
        table_with_col = [W_expanded[tple] for tple in W_expanded if tple[0] == col][0]
        col_counts = np.sum(table_with_col, axis=1)
        W_expanded[col, col] = np.diag(col_counts)

    return W_expanded


def one_hot_encode(df, features_to_encode, attribute_dict):
    """
    Performs one-hot encoding of the data as described in the paper
    """
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
