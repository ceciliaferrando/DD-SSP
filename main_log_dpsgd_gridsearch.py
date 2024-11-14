import argparse
import dill
import sys

sys.path.append('..')
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from ddssp import *
from utils import *
from baselines import *
from DPSGDlogreg import *
#from dpsgd_opacus import *
from log_dpsgd import *

parser = argparse.ArgumentParser(description='Experiment Inputs')
parser.add_argument('--dataset', help='Dataset', type=str, default='adult')
parser.add_argument('--method', help='Method to be used', type=str, default='dpsgd',
                    choices=['public', 'genobjpert', 'aim'])
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--num_experiments', type=int, default=1)
parser.add_argument('--seed', type=int, default=242)
parser.add_argument('--n_limit', type=int, default=50_000)
parser.add_argument('--one_hot', type=str, default='True')
parser.add_argument('--scale_y', type=str, default='False')
parser.add_argument('--aim_y_mrg_opt', type=str, default='False')
parser.add_argument('--grad_clip_norm', type=float, default=1.0)

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
grad_clip_norm = args.grad_clip_norm

if __name__ == "__main__":

    ##### AIM model parameters ########################################################################################
    model_size = 200
    max_iters = 1000
    PGMmarginals = 'all-pairs'

    ##### Setup ########################################################################################################

    np.random.seed(seed)

    target_dict = {'adult': 'income>50K', 'ACSincome': 'PINCP', 'ACSemployment': 'ESR', "ACSmobility": 'MIG',
                   "ACSPublicCoverage": 'PUBCOV', 'ACSTravelTime': 'JWMNP', 'logregbinary20': 'predicted'}

    (X, X_test, y, y_test, X_pre, X_test_pre,
     pgm_train_df, domain, target,
     attribute_dict, features_to_encode, encoded_features,
     original_ranges, all_columns, zero_std_cols) = preprocess_data(dataset, target_dict, n_limit,
                                                                    one_hot, scale_y)

    n, d = X.shape

    res_out = []
    col_out = ['dataset', 'method', 'auc', 'experiment_n', 'seed', 'n_limit',
               'param', 'epsilon', 'delta', 'batch_size', 'learning_rate', 'grad_clip_norm', 'epochs']
    outdir = dataset + "_logreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    models_dir = "models_logistic"

    ##### Grid Search Hyperparameters #################################################################################

    batch_sizes = [len(X)]
    grad_clip_norms = [0.2] # best clipnorm nondp: 0.2
    epochs_list = [200]
    learning_rates = [0.005]   # best lr nondp: 0.01
    #learning_rates = [0.0006, 0.0008, 0.001, 0.002, 0.004]

    best_results = {}

    # initialize best AUC variables
    best_non_dp_auc = 0
    best_dp_auc = 0

    ##### Experiment over t trials #####################################################################################
    for t in range(num_experiments):

        # for epsilon in [50]:
        for epsilon in [50]:

            print(f'epsilon: {epsilon}, t: {t}')

            ##### bounds ###############################################################################################
            bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, one_hot)
            bound_y = 1 if one_hot else np.abs(np.max(attribute_dict[target]))

            ##### Grid Search Loop for DPSGD ##########################################################################
            if method == 'dpsgd':

                for batch_size in batch_sizes:
                    for grad_clip_norm in grad_clip_norms:
                        for epochs in epochs_list:
                            for lr in learning_rates:

                                # prepare data loaders
                                train_loader, test_loader = prepare_data_loaders(X.to_numpy(), y.to_numpy().flatten(),
                                                                                 X_test.to_numpy(),
                                                                                 y_test.to_numpy().flatten(),
                                                                                 batch_size=batch_size,
                                                                                 scale_data=False,
                                                                                 shuffle=False)

                                # print(X)
                                # for data, target in train_loader:
                                #     data_np = data.numpy()
                                #     print(data)
                                #
                                # print(X_test)
                                # for data, target in test_loader:
                                #     data_np = data.numpy()
                                #     print(data)

                                # Non-Private Logistic Regression
                                print("Training Non-Private Logistic Regression...")
                                non_private_model = LogisticRegressionModel(input_dim=d)
                                non_dp_trainer = NonPrivateSGDTrainer(
                                    model=non_private_model,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    epochs=epochs,
                                    lr=lr
                                )
                                non_dp_trainer.train()
                                non_dp_trainer.evaluate()
                                non_dp_auc = non_dp_trainer.auc()
                                non_dp_trainer.plot_loss()
                                non_dp_trainer.plot_grad_norms()

                                non_dp_trainer.compare_loss_with_custom(X, y)

                                print(stop)

                                # record non-DP results
                                res_out.append(
                                    [dataset, 'non-dpsgd', non_dp_auc, t, seed, n_limit, None, epsilon, delta,
                                     batch_size,
                                     lr, grad_clip_norm, epochs]
                                )

                                # Update best non-DP AUC if current is higher
                                if non_dp_auc > best_non_dp_auc:
                                    best_non_dp_auc = non_dp_auc

                                # Differentially Private Logistic Regression
                                print("Training Differentially Private Logistic Regression...")

                                dp_model = LogisticRegressionModel(input_dim=d)
                                dp_trainer = DPSGDTrainer(
                                    model=dp_model,
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    epochs=epochs,
                                    lr=lr,
                                    epsilon=epsilon,
                                    delta=delta,
                                    max_grad_norm=grad_clip_norm,
                                    use_dp=True  # Set to True to enable DP
                                )
                                dp_trainer.train()
                                dp_trainer.evaluate()
                                dpsgd_auc = dp_trainer.auc()
                                dp_trainer.plot_loss()
                                dp_trainer.plot_grad_norms()

                                # record DP results
                                res_out.append(
                                    [dataset, 'dpsgd', dpsgd_auc, t, seed, n_limit, None, epsilon, delta, batch_size,
                                     lr, grad_clip_norm, epochs]
                                )

                                # Update best DP AUC if current is higher
                                if dpsgd_auc > best_dp_auc:
                                    best_dp_auc = dpsgd_auc

            ##### Save the results progressively ######################################################################
            out_df = pd.DataFrame(res_out, columns=col_out)
            one_hot_flag_str = '' if not one_hot else 'one-hot_True'
            filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                           f'{seed}seed_{one_hot_flag_str}onehot.csv'
            out_df.to_csv(filename_out, index=False)

        # After all experiments, print all settings and corresponding AUCs
        print("All settings and their corresponding AUCs:")
        for result in res_out:
            print(
                f"Settings -> Method: {result[1]}, Batch Size: {result[9]}, Learning Rate: {result[10]}, "
                f"Grad Clip Norm: {result[11]}, Epochs: {result[12]}, Epsilon: {result[7]} -> AUC: {result[2]}"
            )

        # Print the best AUCs for non-DP and DP methods
        print(f"\nBest Non-DP AUC: {best_non_dp_auc}")
        print(f"Best DP AUC: {best_dp_auc}")

        # Optionally, save all results to a JSON file
        all_results_out = {
            f"{result[1]}_{result[9]}_{result[10]}_{result[11]}_{result[12]}_{result[7]}": result[2] for
            result in res_out
        }

        all_results_filename = f'{outdir}all_results_{dataset}_{method}_{num_experiments}exps_{n_limit}limit_{seed}seed.json'
        with open(all_results_filename, 'w') as f:
            json.dump(all_results_out, f, indent=4)