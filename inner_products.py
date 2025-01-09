import argparse
import dill
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from ddssp import *
from utils import *
from baselines import *
from DPSGDlogreg import *
from dpsgd_opacus import *

parser = argparse.ArgumentParser(description='Experiment Inputs')
parser.add_argument('--dataset', help='Dataset', type=str, default='ACSemployment')
parser.add_argument('--method', help='Method to be used', type=str, default='public',
                    choices=['public', 'genobjpert', 'aim'])
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--num_experiments', type=int, default=5)
parser.add_argument('--seed', type=int, default=242)
parser.add_argument('--n_limit', type=int, default=50_000)
parser.add_argument('--one_hot', type=str, default='True')
parser.add_argument('--scale_y', type=str, default='False')
parser.add_argument('--aim_y_mrg_opt', type=str, default='False')

args = parser.parse_args()

dataset = args.dataset
method = args.method
delta = args.delta
rho = 0.05
num_experiments = args.num_experiments
seed = args.seed
n_limit = args.n_limit
one_hot = True if args.one_hot == 'True' else False
scale_y = False
aim_y_mrg_opt = True if args.aim_y_mrg_opt == 'True' else None

if __name__ == "__main__":

    ##### AIM model parameters  ########################################################################################
    model_size = 200
    max_iters = 1000
    PGMmarginals = 'all-pairs'

    ##### Setup ########################################################################################################

    np.random.seed(seed)

    target_dict = {'adult': 'income>50K', 'ACSincome': 'PINCP', 'ACSemployment': 'ESR', "ACSmobility": 'MIG',
                   "ACSPublicCoverage": 'PUBCOV', 'ACSTravelTime': 'JWMNP'}

    (X, X_test, y, y_test, X_pre, X_test_pre,
     pgm_train_df, domain, target,
     attribute_dict, features_to_encode, encoded_features,
     original_ranges, all_columns, zero_std_cols) = preprocess_data(dataset, target_dict, n_limit,
                                                                    one_hot, scale_y)

    n, d = X.shape

    print(X.columns)
    print(X_test.columns)

    res_out = []
    col_out = ['dataset', 'method', 'auc', 'experiment_n', 'seed', 'n_limit',
               'param', 'epsilon', 'delta']
    outdir = dataset + "_logreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    models_dir = "models_logistic"

    ##### Experiment over t trials #####################################################################################

    for t in range(num_experiments):

        #for epsilon in [0.05, 0.1, 0.5, 1.0, 2.0]:
        #for epsilon in [0.5, 1.0, 2.0, 5.0]:
        for epsilon in [1]:

            print(f'epsilon: {epsilon}, t: {t}')

            ##### bounds ###############################################################################################
            bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, one_hot)
            bound_y = 1 if one_hot else np.abs(np.max(attribute_dict[target]))

            ##### public ###############################################################################################

            def public_logreg_regularized(X, y, all_columns):
                logreg = LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=2000)
                logreg.fit(X, y)
                return logreg.coef_.flatten()

            theta_public = public_logreg_regularized(X, y, all_columns)

            public_auc = testLogReg(theta_public, X_test, y_test)

            res_out.append([dataset, method, public_auc, t, seed, n_limit, None, epsilon, delta])
            print(f"public auc reg: {public_auc}")

            inner_products = np.zeros((len(y),))
            y_enc = y * 2 - 1
            print(X)
            print(y_enc)
            for i in range(len(y_enc)):
                ip = np.dot(y_enc.to_numpy()[i] * X.to_numpy()[i,:], theta_public)
                inner_products[i,] = ip

            np.save(f'{dataset}_innerprods.npy', inner_products)

            print("DONE")
            print(stop)

            # inner_products = np.dot(X.to_numpy() * y.to_numpy()[:, np.newaxis], theta_public)

            # Display some statistics about the inner products
            min_inner_product = inner_products.min()
            max_inner_product = inner_products.max()
            mean_inner_product = inner_products.mean()
            percentile_10 = np.percentile(inner_products, 10)
            percentile_90 = np.percentile(inner_products, 90)

            print(
                f"Inner product (y_n * x_n * theta_public) min: {min_inner_product}, "
                f"max: {max_inner_product}, mean: {mean_inner_product}, "
                f"10th percentile: {percentile_10}, 90th percentile: {percentile_90}"
            )
