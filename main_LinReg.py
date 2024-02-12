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
    parser.add_argument('--method', help='Method to be used', type=str, default='adassp',
                        choices=['public', 'adassp', 'aim'])
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--num_experiments', type=int, default=5)
    parser.add_argument('--seed', type=int, default=238)
    parser.add_argument('--n_limit', type=int, default=20_000)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--one_hot', type=str, default='True')
    args = parser.parse_args()

    dataset = args.dataset
    method = args.method
    delta = args.delta
    num_experiments = args.num_experiments
    seed = args.seed
    n_limit = args.n_limit
    train_ratio = args.train_ratio
    one_hot = True if args.one_hot == 'True' else False
    rescale = one_hot # if one hot encoding is active, always rescale features

    marginals_pgm = 'all-pairs'
    target_dict = {'adult': 'education-num', 'ACSincome-LIN': 'PINCP',
                   'ACSPublicCoverage': 'AGEP', 'ACSmobility': 'AGEP', 'linregbinary': 'predicted',
                   'linregbinary10': 'predicted',
                   'linregbinary30': 'predicted'}
    delta = delta
    epsilon_vec = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    one_hot = one_hot
    printbool = True

    model_size = 40
    max_iters = 1000

    ##### DATA SETUP #############################################################################################
    np.random.seed(seed)

    (X, X_test, y, y_test, pgm_train_df, domain, target, attribute_dict,
     cols_to_dummy, encoded_features, original_ranges, training_columns) = preprocess_data(dataset, target_dict,
                                                                         n_limit, train_ratio, one_hot)


    # print("X", X)
    # print("\n")
    # print("y", y)
    # print("\n")
    # print("X_test", X_test)
    # print("TARGET:", target)


    # Set up output and pbar
    res_out = []
    col_out = ['dataset', 'method', 'mse', 'r2', 'experiment_n', 'seed', 'n_limit', 'train_ratio',
               'param', 'epsilon', 'delta']
    outdir = dataset + "_linreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pbar = tqdm(total=len(epsilon_vec) * num_experiments,
                desc=f'Running {num_experiments} for {method}.')

    for epsilon in epsilon_vec:
        for t in range(num_experiments):
            np.random.seed(seed + t)

            if method == 'public':

                theta_public, mse_public, r2_public = public_method(X, y, X_test, y_test)

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

                theta_aim, mse_aim, r2_aim = dp_query_ss_method(W_expanded, attribute_dict, training_columns, target,
                                                                cols_to_dummy, one_hot, X_test, y_test)

                print(f"mse_aim, {mse_aim}")
                res_out.append([dataset, "aim-ss", mse_aim, r2_aim, t, seed, n_limit, train_ratio, None, epsilon, delta])
                pbar.update(1)


                ####### AIM SYNTHETIC DATA
                theta_aim_synth, mse_aim_synth, r2_aim_synth = dp_query_synth_data_method(synth_X, synth_y,
                                                                                          training_columns, cols_to_dummy,
                                                                                          attribute_dict, encoded_features,
                                                                                          X_test, y_test, one_hot)

                print(f"mse_aim_synth, {mse_aim_synth}")
                res_out.append([dataset, method, mse_aim_synth, r2_aim_synth, t, seed,
                                n_limit, train_ratio, None, epsilon, delta])
                pbar.update(1)

            elif method == 'adassp':

                rho = 0.05

                bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, cols_to_dummy, one_hot)
                bound_y = 1 if one_hot else np.abs(np.max(attribute_dict[target]))
                print(f"bound_y {bound_y}")
                print(f"bound_XTX {bound_XTX}")
                print(f"bound_X {bound_X}")

                # adassp method
                theta_adassp, mse_adassp, r2_adassp = AdaSSP_linear_regression(X, y, epsilon, delta,
                                                                               rho, bound_X, bound_y, bound_XTX,
                                                                               X_test, y_test)

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