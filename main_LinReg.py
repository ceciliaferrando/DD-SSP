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

from methods_LinReg import *
from utils import *
from utils_LinReg import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--dataset', help='Dataset', type=str, default='ACSincome-LIN')
    parser.add_argument('--method', help='Method to be used', type=str, default='aim',
                        choices=['public', 'adassp', 'aim'])
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--num_experiments', type=int, default=1)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--n_limit', type=int, default=20_000)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--iterate_over_gamma', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--one_hot', action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    dataset = args.dataset
    method = args.method
    delta = args.delta
    num_experiments = args.num_experiments
    seed = args.seed
    n_limit = args.n_limit
    train_ratio = args.train_ratio
    one_hot = args.one_hot
    rescale = True

    marginals_pgm = 'all-pairs'
    target_dict = {'adult': 'education-num', 'ACSincome-LIN': 'PINCP',
                   'ACSPublicCoverage': 'AGEP', 'ACSmobility': 'AGEP', 'linregbinary': 'predicted',
                   'linregbinary10': 'predicted',
                   'linregbinary30': 'predicted'}
    delta = delta
    epsilon_vec = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    one_hot = one_hot
    printbool = True
    np.random.seed(seed)

    model_size = 40
    max_iters = 1000

    ##### DATA SETUP #############################################################################################

    # Import the data
    csv_path = '../hd-datasets-master/clean/' + dataset + '.csv'
    meta_path = '../hd-datasets-master/clean/' + dataset + '-domain.json'
    data = Dataset.load(csv_path, meta_path)  # for PGM
    domain = data.domain
    target = target_dict[dataset]

    # Create the dataset to save the results
    outdir = dataset + "_linreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load the data
    df = pd.read_csv(csv_path)
    # features = df.drop(target, inplace=False, axis=1).columns

    # Set up output and pbar
    res_out = []
    col_out = ['dataset', 'method', 'mse', 'r2', 'experiment_n', 'seed', 'n_limit', 'train_ratio',
               'param', 'epsilon', 'delta']

    ##### EXPERIMENT #############################################################################################

    pbar = tqdm(total=len(epsilon_vec) * num_experiments,
                desc=f'Running {num_experiments} for {method}.')

    # Split the DF
    # X, y, X_test and y_test are all Pandas dataframes
    if len(df) > n_limit:
        df = df[:n_limit]
    X, y, X_test, y_test = splitdf(df, target, train_ratio)
    pgm_train_df = X.join(y)

    # Create dictionary with attribute levels
    attribute_dict = {}
    for col in df:
        unique_values = list(range(domain[col]))
        attribute_dict[col] = unique_values
    print(attribute_dict)

    # If one_hot is active, then we one hot both the train set and the test set.
    # Here we just define the variables we will use later.
    cols_to_dummy, training_columns = [], None
    if one_hot == True:
        print("one-hot encoding...")
        # We find which columns we need to one-hot encode. The others we don't encode are categorical ordinal
        cols_to_dummy = get_cols_to_dummy(dataset)

        # We first one-hot encode the training data. We drop the first column in Pandas to avoid multi-collinearity
        # issues in the train data. We then check whether there is a column with all 0 or all 1 and we remove it.
        # The trick by which we remove the columns with the same value is to see which columns have a standard
        # deviation equal to 0 or not
        X_ohe = pd.get_dummies(X, columns=cols_to_dummy, drop_first=True)
        X_ohe.drop(X_ohe.std()[X_ohe.std() == 0].index, axis=1, inplace=True)
        for col in X_ohe:
            if col.endswith(".0"):
                X_ohe.rename(columns={col: col.split(".0")[0]}, inplace=True)

        training_columns = X_ohe.columns

        # Now we do the one-hot encoding for the test set. Once we do the one-hot encoding:
        # (1) we need to remove the columns that are not in the training columns
        # (2) we need to add with all zeros the columns that are in the training data but not in the test data
        X_test_ohe = pd.get_dummies(X_test, columns=cols_to_dummy, drop_first=True)
        X_test_ohe = add_and_subsets_cols_to_test_set(X_test_ohe, training_columns)
        for col in X_test_ohe:
            if col.endswith(".0"):
                X_test_ohe.rename(columns={col: col.split(".0")[0]}, inplace=True)

        assert set(X_ohe.columns) == set(X_test_ohe.columns)

        X = X_ohe.copy(deep=True)
        print("X after 1h:", X)
        X_test = X_test_ohe.copy()

    # useful for encoding
    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
    original_ranges = {feature: [0, domain[feature]] for feature in attribute_dict.keys()}

    # Run safety check for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature to check for multicollinearity
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]

    print(f"VIF for X: {vif_data}")

    for epsilon in epsilon_vec:
        for t in range(num_experiments):
            np.random.seed(seed + t)

            if method == 'public':

                theta_public, mse_public, r2_public = public_method(X, y, X_test, y_test,
                                                                    cols_to_dummy, attribute_dict, one_hot)

                res_out.append([dataset, method, mse_public, r2_public, t, seed,
                                n_limit, train_ratio, None, epsilon, delta])

                print(f"mse_public, {mse_public}")
                pbar.update(1)

            elif method == 'aim':
                # AIM sufficient statistics method
                # Instantiate and train pgm synthesizer (AIM), generate synthetic data

                n, d = X.shape

                # get marginal tables and synthetic data for training
                W, synth_X, synth_y = get_aim_W_and_data(pgm_train_df, domain, target, marginals_pgm,
                                                         epsilon, delta, model_size, max_iters, n)
                W_expanded = expand_W(W, attribute_dict)
                W_filename = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                             f'{seed}seed_W.pickle'
                with open(W_filename, 'wb') as handle:
                    pickle.dump(W, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(W_filename, 'rb') as handle:
                    W_load = pickle.load(handle)

                encoded_features = [col.split(".")[0] for col in X if col.split("_")[0] in cols_to_dummy]

                theta_aim, mse_aim, r2_aim = dp_query_ss_method(W_expanded, attribute_dict, training_columns,
                                                       encoded_features, target, domain, n,
                                                        cols_to_dummy, one_hot, X_test, y_test)

                res_out.append([dataset, "aim-ss", mse_aim, r2_aim, t, seed, n_limit, train_ratio, None, epsilon, delta])
                pbar.update(1)


                ####### AIM SYNTHETIC DATA
                theta_aim_synth, mse_aim_synth, r2_aim_synth = dp_query_synth_data_method(synth_X, synth_y,
                                                                    training_columns, attribute_dict, cols_to_dummy,
                                                                    encoded_features, X_test, y_test, one_hot)

                print(f"mse_aim_synth, {mse_aim_synth}")
                res_out.append([dataset, method, mse_aim_synth, r2_aim_synth, t, seed,
                                n_limit, train_ratio, None, epsilon, delta])
                pbar.update(1)

            elif method == 'adassp':

                rho = 0.05

                if rescale == True:
                    # normalize numerical features between -1 and +1
                    # making sure to exclude the one-hot encoded categorical variables, which should be in domain [0, 1]
                    # rescale target too
                    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
                    original_ranges = {feature: [0, domain[feature]] for feature in attribute_dict.keys()}
                    X_adassp = normalize_minus1_1(X, encoded_features, original_ranges)
                    X_test_adassp = normalize_minus1_1(X_test, encoded_features, original_ranges)
                    y_adassp = normalize_minus1_1(y, encoded_features, original_ranges)
                else:
                    X_adassp = copy.deepcopy(X)
                    X_test_adassp = copy.deepcopy(X_test)
                    y_adassp = copy.deepcopy(y)

                bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, cols_to_dummy, one_hot, rescale)
                bound_y = np.abs(np.max(attribute_dict[target]))
                print(f"bound_y {bound_y}")
                print(f"bound_XTX {bound_XTX}")
                print(f"bound_X {bound_X}")

                # adassp method
                theta_adassp, mse_adassp, r2_adassp = AdaSSP_linear_regression(X_adassp, y_adassp, epsilon, delta,
                                                                               rho, bound_X, bound_y, bound_XTX,
                                                                               X_test_adassp, y_test,
                                                                               [min(attribute_dict[target]),
                                                                                max(attribute_dict[target])],
                                                                               rescale=rescale)

                print(f"theta adassp: {theta_adassp}")
                print("mse adassp", mse_adassp)
                res_out.append([dataset, method, mse_adassp, r2_adassp, t, seed,
                                n_limit, train_ratio, None, epsilon, delta])
                pbar.update(1)

            else:
                raise NotImplementedError('Method not implemented yet.')

            out_df = pd.DataFrame(res_out, columns=col_out)
            one_hot_flag_str = '' if not one_hot else 'one-hot_True'
            filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                           f'{seed}seed_{one_hot_flag_str}.csv'
            out_df.to_csv(filename_out)