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
import pickle

from utils import *
from utils_LogReg import *
from methods_LogReg import *
from dpquery_chebyshev import Chebyshev, phi_logit


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--dataset', help='Dataset', type=str, default='adult')
    parser.add_argument('--method', help='Method to be used', type=str, default='public',
                        choices=['public', 'diffprivlib', 'aim', 'genobjpert'])
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--num_experiments', type=int, default=1)
    parser.add_argument('--seed', type=int, default=236)
    parser.add_argument('--n_limit', type=int, default=20_000)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--iterate_over_gamma', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--one_hot', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--rescale', action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    
    
    dataset = args.dataset
    method = args.method
    delta = args.delta
    num_experiments = args.num_experiments
    seed = args.seed
    n_limit = args.n_limit
    train_ratio = args.train_ratio
    iterate_over_gamma = args.iterate_over_gamma
    one_hot = args.one_hot
    rescale = args.rescale
    print(f"one hot {one_hot}, rescale {rescale}")
    
    model_size = 40  # FOR AIM
    max_iters = 1000   # FOR AIM

    # Setup hyper-parameters for the runs
    PGMmarginals = 'all-pairs'
    QUAILmarginals = 'no-target-pairs'
    target_dict = {'adult': 'income>50K', 'titanic': 'Survived', 'diabetes': 'Outcome',
                   'ACSemployment': 'ESR', 'ACSincome': 'PINCP', "ACSmobility": 'MIG',
                   "ACSPublicCoverage": 'PUBCOV', 'ACSTravelTime': 'JWMNP', 'logregbinary10': 'predicted',
                   'logregbinary5lowcorr': 'predicted', 'logregbinary7lowcorr': 'predicted',
                   'logregbinary10lowcorr': 'predicted', 'logregbinary20lowcorr': 'predicted',
                   'logregbinary30lowcorr': 'predicted'}

    epsilon_vec = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    np.random.seed(seed=seed)

    # Import the data
    csv_path = '../hd-datasets-master/clean/' + dataset + '.csv'
    meta_path = '../hd-datasets-master/clean/' + dataset + '-domain.json'
    data = Dataset.load(csv_path, meta_path)  # for PGM
    domain = data.domain
    target = target_dict[dataset]
    printbool = True
    marginals_pgm = PGMmarginals  # "" (for MST), "target-pairs", "target-triplets", "all-triplets" (for MWEM)
    marginals_QUAIL = QUAILmarginals

    # Create the dataset to save the results
    outdir = dataset + "_logreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load the data
    df = pd.read_csv(csv_path)
    features = df.drop(target, inplace=False, axis=1).columns

    res_out = []
    col_out = ['dataset', 'method', 'auc', 'f1', 'accuracy', 'experiment_n', 'seed', 'n_limit', 'train_ratio',
               'param', 'epsilon']
    pbar = tqdm(total=len(epsilon_vec) * num_experiments,
                desc=f'Running {num_experiments} for {method}.')

    # Split the DF
    # X, y, X_test and y_test are all Pandas dataframes
    if len(df) > n_limit:
        df = df[:n_limit]
    X, y, X_test, y_test = splitdf(df, target, train_ratio)
    pgm_train_df = X.join(y)
    print("y", y)

    # Create dictionary with feature levels
    feature_dict = {}
    for column in X.columns:
        unique_values = sorted(X[column].unique().tolist())  #### HERE ORDER MATTERS
        feature_dict[column] = unique_values
    features = list(feature_dict.keys())
    target_values = sorted(y[target].unique().tolist())

    attribute_dict = {}
    for col in df:
        unique_values = list(range(domain[col]))
        attribute_dict[col] = unique_values
    attribute_dict[target] = [-1, 1]
    print(attribute_dict)

    # If one_hot is active, then we one hot both the train set and the test set.
    # Here we just define the variables we will use later.
    cols_to_dummy, training_columns = None, X.columns
    if one_hot:
        # We find which columns we need to one-hot encode. The others we don't encode are categorical ordinal
        cols_to_dummy = get_cols_to_dummy(dataset)

        # We first one-hot encode the training data. We drop the first column in Pandas to avoid multi-collinearity
        # issues in the train data. We then check whether there is a column with all 0 or all 1 and we remove it.
        # The trick by which we remove the columns with the same value is to see which columns have a standard
        # deviation equal to 0 or not
        X_ohe = pd.get_dummies(X, columns=cols_to_dummy, drop_first=True)
        X_ohe.drop(X_ohe.std()[X_ohe.std() == 0].index, axis=1, inplace=True)
        training_columns = X_ohe.columns
        print(training_columns)

        # Now we do the one-hot encoding for the test set. Once we do the one-hot encoding:
        # (1) we need to remove the columns that are not in the training columns
        # (2) we need to add with all zeros the columns that are in the training data but not in the test data
        X_test_ohe = pd.get_dummies(X_test, columns=cols_to_dummy, drop_first=True)
        X_test_ohe = add_and_subsets_cols_to_test_set(X_test_ohe, training_columns)

        assert set(X_ohe.columns) == set(X_test_ohe.columns)

        X = X_ohe.copy()
        X_test = X_test_ohe.copy()

    # print(f'Max Sensitivity: {max([np.linalg.norm(row) for row in X.to_numpy()])}')

    for t in range(num_experiments):
        np.random.seed(seed + t)
        for epsilon in epsilon_vec:
            print("EPSILON =", epsilon)

            if method == 'public':

                (public_f1score, public_accuracy,
                 public_fpr, public_tpr,
                 public_threshold, public_auc) = public_method(X, y, X_test, y_test,
                                                               cols_to_dummy, attribute_dict, one_hot, approx = False)
                res_out.append([dataset, method, public_auc, public_f1score, public_accuracy, t, seed,
                                n_limit, train_ratio, None, epsilon])
                print(f"public auc reg: {public_auc}")
                pbar.update(1)

            elif method == 'public-approx-ss':
                (public_f1score, public_accuracy,
                 public_fpr, public_tpr,
                 public_threshold, public_auc) = public_method(X, y, X_test, y_test,
                                                               cols_to_dummy, attribute_dict, one_hot, approx = True)
                res_out.append([dataset, method, public_auc, public_f1score, public_accuracy, t, seed,
                                n_limit, train_ratio, None, epsilon])
                print(f"public auc reg: {public_auc}")
                pbar.update(1)

            elif method == 'aim':
                # add the same regularization as genobjpert
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
                # filename = "/Users/ceciliaferrando/Documents/UMASS/RESEARCH/DPsynthesisML/dp-reg/adult_logreg/W.pickle"
                # with open(filename, 'rb') as handle:
                #     W = pickle.load(handle)

                ######## AIM SS #########
                encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]

                (DPapprox_f1score, DPapprox_accuracy,
                 DPapprox_fpr, DPapprox_tpr,
                 DPapprox_threshold, DPapprox_auc) = dp_query_approx_ss_method(W_expanded, attribute_dict, training_columns,
                                                                               encoded_features, target,
                                                                               domain, n, cols_to_dummy, one_hot,
                                                                               X_test, y_test)
                res_out.append([dataset, 'aim-ss', DPapprox_auc, DPapprox_f1score, DPapprox_accuracy, t,
                                seed, n_limit, train_ratio, None, epsilon])
                print("DPapprox_auc", DPapprox_auc)

                ######## AIM Synthetic Data #########

                (aimsynth_f1score, aimsynth_accuracy,
                 aimsynth_fpr, aimsynth_tpr,
                 aimsynth_threshold, aimsynth_auc) = dp_query_synth_data_method(synth_X, synth_y,
                                                                                training_columns, attribute_dict,
                                                                                cols_to_dummy, encoded_features,
                                                                                X_test, y_test, one_hot)

                res_out.append([dataset, method, aimsynth_auc, aimsynth_f1score, aimsynth_accuracy, t,
                                seed, n_limit, train_ratio, None, epsilon])
                print(f"aim synth auc: {aimsynth_auc}")

                pbar.update(1)

            elif method == 'genobjpert':
                # Generalizeed Obj Perturbation method

                genobj_auc = objective_perturbation_method(X, y, X_test, y_test, attribute_dict, target,
                                                           cols_to_dummy, epsilon, delta, one_hot)

                res_out.append([dataset, method, genobj_auc, None, None, t,
                                seed, n_limit, train_ratio, None, epsilon])
                print(f"genobjpert auc reg: {genobj_auc}")
                pbar.update(1)

            else:
                raise NotImplementedError('Method not implemented yet.')

            out_df = pd.DataFrame(res_out, columns=col_out)
            todays_date = datetime.now().strftime('%Y-%m-%d-%H-%M')
            filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                           f'{seed}seed_{todays_date}.csv'
            out_df.to_csv(filename_out)