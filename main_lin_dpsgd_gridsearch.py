import argparse
import sys
import json
from itertools import product

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
from dpsgd_models import LogisticRegressionModel, LinearRegressionModel
from dpsgd_dp_trainer import DPSGDTrainer
from dpsgd_utils import prepare_data_loaders

parser = argparse.ArgumentParser(description='Experiment Inputs')
parser.add_argument('--dataset', help='Dataset', type=str, default='adult')
parser.add_argument('--method', help='Method to be used', type=str, default='aim',
                    choices=['public', 'adassp', 'aim'])
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--num_experiments', type=int, default=5)
parser.add_argument('--seed', type=int, default=242)
parser.add_argument('--n_limit', type=int, default=50000)
parser.add_argument('--one_hot', type=str, default='True')
parser.add_argument('--scale_y', type=str, default='True')
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
scale_y = True if args.scale_y == 'True' else False
aim_y_mrg_opt = True if args.aim_y_mrg_opt == 'True' else None

if __name__ == "__main__":

    ##### AIM model parameters  ########################################################################################
    model_size = 200
    max_iters = 1000
    PGMmarginals = 'all-pairs'

    ##### Setup  #######################################################################################################

    np.random.seed(seed)

    target_dict = {'adult': 'education-num', 'taxi': 'totalamount', 'fire': 'Priority', 'ACSincome-LIN': 'PINCP',
                   'ACSPublicCoverage': 'AGEP', 'ACSmobility': 'AGEP', 'linregbinary': 'predicted',
                   'linregbinary10': 'predicted',
                   'linregbinary30': 'predicted'}

    (X, X_test, y, y_test, X_pre, X_test_pre,
     pgm_train_df, domain, target,
     attribute_dict, features_to_encode, encoded_features,
     original_ranges, all_columns, zero_std_cols) = preprocess_data(dataset, target_dict, n_limit,
                                                                    one_hot, scale_y)

    n, d = X.shape

    ##### Grid Search Hyperparameters #################################################################################
    batch_sizes = [len(X), 1024, 256]
    grad_clip_norms = [0.01, 0.1, 0.2]
    epochs_list = [1, 10, 20]
    learning_rates = [0.001, 0.01, 0.1, 1.0]

    hyperparam_space = list(product(batch_sizes, grad_clip_norms, epochs_list, learning_rates))
    use_dp = True

    epsilon_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    ##### Check for Saved Best Hyperparameters ########################################################################
    outdir = dataset + "_linreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    best_params_per_epsilon = {}

    #rdp_alphas_for_epsilon = {0.05: [200]}

    for epsilon in epsilon_values:

        # get effective budget

        #rdp_alphas = rdp_alphas_for_epsilon[epsilon]
        rdp_alphas = None

        best_params_file = os.path.join(f"0108_lin_best_hyperparams_{dataset}_{epsilon}.json")
        if os.path.exists(best_params_file):
            print(f"Loading best hyperparameters from {best_params_file}")
            with open(best_params_file, "r") as f:
                best_params_per_epsilon = json.load(f)
        else:
            print("No saved best hyperparameters found. Running grid search...")
            ##### Grid Search for Best Hyperparameters ####################################################################

            print(f"Running grid search for epsilon={epsilon}")
            best_mse = float("inf")
            best_params = None

            for batch_size in batch_sizes:
                for grad_clip_norm in grad_clip_norms:
                    for epochs in epochs_list:
                        for lr in learning_rates:
                            train_loader, test_loader = prepare_data_loaders(
                                X.to_numpy(), y.to_numpy().flatten(),
                                X_test.to_numpy(), y_test.to_numpy().flatten(),
                                batch_size=batch_size, scale_data=False, shuffle=False)

                            dp_model = LinearRegressionModel(input_dim=d)
                            dp_trainer = DPSGDTrainer(
                                model=dp_model,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                epochs=epochs,
                                lr=lr,
                                epsilon=epsilon,
                                delta=delta,
                                max_grad_norm=grad_clip_norm,
                                use_dp=use_dp,
                                rdp_alphas=rdp_alphas
                            )

                            print("smallest alpha", dp_trainer.smallest_alpha)
                            dp_trainer.train()
                            dpsgd_mse = dp_trainer.mse()

                            if dpsgd_mse < best_mse:
                                best_mse = dpsgd_mse
                                best_params = {
                                    'Batch_Size': batch_size,
                                    "Grad_Clip_Norm": grad_clip_norm,
                                    "Epochs": epochs,
                                    "Learning_Rate": lr
                                }

            best_params_per_epsilon[str(epsilon)] = best_params
            print(f"Best parameters for epsilon={epsilon}: {best_params} with MSE={best_mse}")

        # Save best hyperparameters to JSON
        with open(best_params_file, "w") as f:
            json.dump(best_params_per_epsilon, f, indent=4)
        print(f"Best hyperparameters saved to {best_params_file}")

    ##### Run Trials with Best Hyperparameters ########################################################################
    for trial in range(num_experiments):  # Number of trials
        trial_res_out = []
        print(f"Starting Trial {trial + 1}")

        for epsilon in epsilon_values:
            print(f"Running trial {trial + 1} for epsilon={epsilon} using best hyperparameters")
            print(best_params_per_epsilon)
            best_params = best_params_per_epsilon[str(epsilon)]

            train_loader, test_loader = prepare_data_loaders(
                X.to_numpy(), y.to_numpy().flatten(),
                X_test.to_numpy(), y_test.to_numpy().flatten(),
                batch_size=best_params['Batch_Size'], scale_data=False, shuffle=True)

            dp_model = LinearRegressionModel(input_dim=d)
            dp_trainer = DPSGDTrainer(
                model=dp_model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=best_params['Epochs'],
                lr=best_params['Learning_Rate'],
                epsilon=epsilon,
                delta=delta,
                max_grad_norm=best_params['Grad_Clip_Norm'],
                use_dp=use_dp,
                rdp_alphas = rdp_alphas
            )
            dp_trainer.train()
            dpsgd_mse = dp_trainer.mse()

            # Append results for this epsilon
            trial_res_out.append([
                dataset, 'dpsgd', dpsgd_mse, trial, seed, n_limit, None, epsilon, delta,
                best_params['Batch_Size'], best_params['Learning_Rate'], best_params['Grad_Clip_Norm'], best_params['Epochs']
            ])

        # Save results for this trial
        col_out = ['dataset', 'method', 'mse', 'experiment_n', 'seed', 'n_limit', 'param', 'epsilon', 'delta',
                   'batch_size', 'learning_rate', 'grad_clip_norm', 'epochs']
        trial_out_df = pd.DataFrame(trial_res_out, columns=col_out)
        one_hot_flag_str = '' if not one_hot else 'one-hot_True'
        trial_filename = f"{outdir}{dataset}_{method}_{epsilon_values[-1]}_{trial}_5exps_{n_limit}limit_{seed}seed_{one_hot_flag_str}.csv"
        trial_out_df.to_csv(trial_filename, index=False)
        print(f"Trial {trial + 1} results saved to {trial_filename}")