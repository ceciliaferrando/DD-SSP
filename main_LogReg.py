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
from dpquery_chebyshev import Chebyshev, phi_logit


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment Inputs')
    parser.add_argument('--dataset', help='Dataset', type=str, default='adult')
    parser.add_argument('--method', help='Method to be used', type=str, default='public-approx-ss',
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
    X_aim = X
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
    cols_to_dummy, training_columns = None, None
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

    for epsilon in epsilon_vec:

        print("EPSILON =", epsilon)

        for t in range(num_experiments):
            np.random.seed(seed + t)

            if method == 'public':
                if one_hot:
                    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
                    original_X_range = {feature: [0, domain[feature]] for feature in feature_dict.keys()}
                    X_pub = normalize_minus1_1(X, encoded_features, original_X_range)
                    X_test_pub = normalize_minus1_1(X_test, encoded_features, original_X_range)
                else:
                    X_pub = copy.deepcopy(X)
                    X_test_pub = copy.deepcopy(X_test)
                # public baseline
                (public_f1score, public_accuracy,
                 public_fpr, public_tpr,
                 public_threshold, public_auc) = testLogReg(X_pub, y, X_test_pub, y_test)
                res_out.append([dataset, method, public_auc, public_f1score, public_accuracy, t, seed,
                                n_limit, train_ratio, None, epsilon])
                print(f"public auc reg: {public_auc}")
                pbar.update(1)

            elif method == 'public-approx-ss':
                if one_hot:
                    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
                    original_X_range = {feature: [0, domain[feature]] for feature in feature_dict.keys()}
                    X_pub = normalize_minus1_1(X, encoded_features, original_X_range)
                    X_test_pub = normalize_minus1_1(X_test, encoded_features, original_X_range)
                else:
                    X_pub = copy.deepcopy(X)
                    X_test_pub = copy.deepcopy(X_test)
                # public approximate SS baseline
                (_, public_approx_accuracy,
                 public_approx_fpr, public_approx_tpr,
                 public_approx_threshold, public_approx_auc) = testApproxSSLogReg(X_pub, y, X_test_pub, y_test)
                res_out.append([dataset, method, public_approx_auc, None, public_approx_accuracy, t,
                                seed, n_limit, train_ratio, None, epsilon])
                pbar.update(1)

            elif method == 'aim':
                # add the same regularization as genobjpert
                n, d = X.shape

                # AIM sufficient statistics method
                # Instantiate and train pgm synthesizer (AIM)
                pgm_train_df = X_aim.join(y)
                print(pgm_train_df)
                pgm_dataset = Dataset(pgm_train_df, domain)
                mrgs = selectTargetMarginals(df.columns, target, mode=marginals_pgm)
                mrgs_wkld = Workload((mrg, sparse.identity) for mrg in mrgs)
                pgm_synth = PGMsynthesizer(pgm_dataset, epsilon, delta, mrgs_wkld, model_size, max_iters, len(X))
                pgm_synth.aimsynthesizer()
                ans_wkld = pgm_synth.ans_wkld
                W = {key: ans_wkld[key].__dict__['values'] for key in ans_wkld}

                W_filename = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                             f'{seed}seed_W.pickle'
                with open(W_filename, 'wb') as handle:
                    pickle.dump(W, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(W_filename, 'rb') as handle:
                    W_load = pickle.load(handle)

                # filename = "/Users/ceciliaferrando/Documents/UMASS/RESEARCH/DPsynthesisML/dp-reg/adult_logreg/W.pickle"
                # with open(filename, 'rb') as handle:
                #     W = pickle.load(handle)

                W_expanded = expand_W(W, attribute_dict)

                ######## AIM SS #########
                all_attributes_expanded = training_columns.append(pd.Index([target]))
                ZTZ = get_ZTZ(W_expanded, attribute_dict, all_attributes_expanded, cols_to_dummy, rescale=rescale)
                XTXy2 = ZTZ.loc[training_columns, training_columns]
                XTy = ZTZ.loc[training_columns, target]

                if one_hot:
                    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
                    original_X_range = {feature: [0, domain[feature]] for feature in feature_dict.keys()}
                    original_y_range = {target: [0, domain[target]]}
                    X_test = normalize_minus1_1(X_test, encoded_features, original_X_range)

                (DPapprox_f1score, DPapprox_accuracy,
                 DPapprox_fpr, DPapprox_tpr,
                 DPapprox_threshold, DPapprox_auc) = testPrivApproxSSLogReg(XTy, XTXy2, X_test, y_test, len(X), C=1.0)
                res_out.append([dataset, 'aim-ss', DPapprox_auc, DPapprox_f1score, DPapprox_accuracy, t,
                                seed, n_limit, train_ratio, None, epsilon])
                print("DPapprox_auc", DPapprox_auc)

                ######## AIM Synthetic Data #########
                # # sample n rows using the synth data mechanism
                synth = pgm_synth.synth.df
                synth_X, synth_y = synth.loc[:, synth.columns != target], synth.loc[:, synth.columns == target]

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
                    print("aim columns", len(aim_columns))

                    # Now we also modify the X_test so that we are sure that X_test also has the same columns as
                    # the synthetic version of the data generated by AIM.
                    X_test_aim = add_and_subsets_cols_to_test_set(X_test, aim_columns)
                    print("aim test columns", len(X_test_aim))
                    synth_X = synth_X_ohe.copy()
                else:
                    X_test_aim = X_test

                synth_X_ordered = pd.DataFrame()

                for col in X_test_aim.columns:
                    synth_X_ordered[col] = synth_X[col]

                if one_hot:
                    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
                    original_X_range = {feature: [0, domain[feature]] for feature in feature_dict.keys()}
                    synth_X_ordered = normalize_minus1_1(synth_X_ordered, encoded_features, original_X_range)
                    X_test_aim = normalize_minus1_1(X_test_aim, encoded_features, original_X_range)

                (aimsynth_f1score, aimsynth_accuracy,
                 aimsynth_fpr, aimsynth_tpr,
                 aimsynth_threshold, aimsynth_auc) = testLogReg(synth_X_ordered, synth_y, X_test_aim, y_test)

                res_out.append([dataset, method, aimsynth_auc, aimsynth_f1score, aimsynth_accuracy, t,
                                seed, n_limit, train_ratio, None, epsilon])
                print(f"aim synth auc: {aimsynth_auc}")

                pbar.update(1)

            elif method == 'amp':
                # AMP method
                amp = ApproximateMinimaPerturbationLR()
                lambda_param = 0.1
                initial_time = time.time()
                theta_amp, _ = amp.run_classification(
                    X.to_numpy(), y.to_numpy(), epsilon, delta, lambda_param,
                    iterate_over_gamma=iterate_over_gamma)

                end_time = time.time() - initial_time
                logits = np.dot(X_test, theta_amp)
                probabilities = 1 / (1 + np.exp(-logits))
                amp_auc = roc_auc_score(y_test, probabilities)
                res_out.append([dataset, method, amp_auc, None, None, t,
                                seed, n_limit, train_ratio, lambda_param, epsilon])
                pbar.update(1)

            elif method == 'genobjpert':
                # Generalizeed Obj Perturbation method

                n, d = X.shape

                print(f"N = {n}, d = {d}")
                print(f"epsilon = {epsilon}")
                print(f"delta = {delta}")

                # maxvals = np.array(np.amax(X.to_numpy(), axis = 0)).reshape(-1,1)
                # max_row_norm = np.sqrt(np.sum([a**2 for a in maxvals]))
                if one_hot:
                    encoded_features = [col for col in X if col.split("_")[0] in cols_to_dummy]
                    original_X_range = {feature: [0, domain[feature]] for feature in feature_dict.keys()}
                    original_y_range = {target: [0, domain[target]]}
                    X_gop = normalize_minus1_1(X, encoded_features, original_X_range)
                    X_test_gop = normalize_minus1_1(X_test, encoded_features, original_X_range)
                    bound_y = np.abs(domain[target])
                    max_row_norm = np.sqrt(len(feature_dict.keys()))
                    print(f"bound_y {bound_y}")
                    print(f"max_row_norm {max_row_norm}")
                else:
                    X_gop = X
                    X_test_gop = X_test
                    max_row_norm = get_bound_X(one_hot, feature_dict, cols_to_dummy, target)
                    # actual_row_max_norm = max([np.linalg.norm(row) for row in X.to_numpy()])
                    # actual_zeta = np.sqrt(n) * actual_row_max_norm

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

                res_out.append([dataset, method, genobj_auc, None, None, t,
                                seed, n_limit, train_ratio, (lmda, zeta), epsilon])
                print(f"genobjpert auc reg: {genobj_auc}")
                pbar.update(1)

            else:
                raise NotImplementedError('Method not implemented yet.')

            out_df = pd.DataFrame(res_out, columns=col_out)
            todays_date = datetime.now().strftime('%Y-%m-%d-%H-%M')
            filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                           f'{seed}seed_{todays_date}.csv'
            out_df.to_csv(filename_out)