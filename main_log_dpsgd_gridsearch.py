import argparse
import dill
import sys

sys.path.append('..')
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from ddssp import *
from utils import *
from baselines import *
from DPSGDlogreg import *
from dpsgd_opacus import *

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

    ##### AIM model parameters  ########################################################################################
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

    batch_sizes = [512, 1024, 2048, 4096]  # Example values
    grad_clip_norms = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Example values
    epochs_list = [1, 2, 5, 10, 20]  # Example values
    learning_rates = [0.001, 0.01, 0.1, 1.0]

    # batch_sizes = [2048]  # Example values
    # grad_clip_norms = [5.0]  # Example values
    # epochs_list = [50]  # Example values
    # learning_rates = [0.001]

    best_results = {}

    ##### Experiment over t trials #####################################################################################
    for t in range(num_experiments):

        # for epsilon in [50]:
        # for epsilon in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        for epsilon in [10, 50, 100]:

            print(f'epsilon: {epsilon}, t: {t}')

            ##### bounds ###############################################################################################
            bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, one_hot)
            bound_y = 1 if one_hot else np.abs(np.max(attribute_dict[target]))

            ##### Grid Search Loop for DPSGD ##########################################################################
            if method == 'dpsgd':

                best_auc = 0.0
                best_params = None

                for batch_size in batch_sizes:
                    for grad_clip_norm in grad_clip_norms:
                        for epochs in epochs_list:
                            for learning_rate in learning_rates:

                                print(
                                    f'Running DPSGD with batch_size={batch_size}, grad_clip_norm={grad_clip_norm}, '
                                    f'learning_rate={learning_rate}, epochs={epochs}')

                                import torch
                                from torch.utils.data import DataLoader, TensorDataset

                                X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
                                y_tensor = torch.tensor(y.to_numpy().flatten(), dtype=torch.float32)
                                train_dataset = TensorDataset(X_tensor, y_tensor)
                                X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
                                y_test_tensor = torch.tensor(y_test.to_numpy().flatten(), dtype=torch.float32)

                                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                                input_dim = d  # Number of features in the dataset
                                model = LogisticRegressionModel(input_dim)

                                print("model initiated, calling function")
                                torch_model = train_dpsgd_opacus(model, train_loader, epochs, epsilon, delta, lr=learning_rate,
                                                                 max_grad_norm=grad_clip_norm)
                                print("done training")

                                torch_model.eval()  # Set model to evaluation mode
                                all_labels = []
                                all_probs = []

                                with torch.no_grad():
                                    for data, labels in test_loader:
                                        outputs = torch_model(data)
                                        probs = outputs.squeeze().cpu().numpy()  # Get predicted probabilities
                                        all_probs.extend(probs)
                                        all_labels.extend(labels.cpu().numpy())

                                # Compute the AUC using sklearn
                                auc_dpsgd = roc_auc_score(all_labels, all_probs)
                                auc_dpsgd_flipped = roc_auc_score(all_labels, 1-np.array(all_probs))

                                auc_dpsgd = max(auc_dpsgd, auc_dpsgd_flipped)
                                print('AUC', auc_dpsgd)

                                res_out.append(
                                    [dataset, method, auc_dpsgd, t, seed, n_limit, None, epsilon, delta, batch_size,
                                     learning_rate, grad_clip_norm, epochs])

                                # Track the best AUC and corresponding parameters
                                if auc_dpsgd > best_auc:
                                    best_auc = auc_dpsgd
                                    best_params = (batch_size, learning_rate, grad_clip_norm, epochs)

                best_results[epsilon] = (best_auc, best_params)
                print(
                    f"Best AUC for epsilon={epsilon}: {best_auc} with batch_size={best_params[0]}, "
                    f"learning_rate={best_params[1]}, grad_clip_norm={best_params[2]}, epochs={best_params[3]}")

            ##### Save the results progressively ######################################################################
            out_df = pd.DataFrame(res_out, columns=col_out)
            one_hot_flag_str = '' if not one_hot else 'one-hot_True'
            filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                           f'{seed}seed_{one_hot_flag_str}onehot.csv'
            out_df.to_csv(filename_out)

    # Print out the best results after all experiments
    print("Best results across all epsilons:")
    for eps, (auc, params) in best_results.items():
        print(
            f"Epsilon: {eps}, Best AUC: {auc}, Batch Size: {params[0]}, Learning Rate: {params[1]}, Grad Clip Norm: {params[2]},  Epochs: {params[3]}")

    # Convert best_results to a JSON-compatible format
    best_results_out = {
        str(eps): {'AUC': auc, 'Batch_Size': params[0], 'Learning Rate': params[1], 'Grad_Clip_Norm': params[2], 'Epochs': params[3]} for
        eps, (auc, params) in best_results.items()}

    # Specify the filename and save the JSON file
    best_results_filename = f'{outdir}best_results_{dataset}_{method}_{num_experiments}exps_{n_limit}limit_{seed}seed.json'
    with open(best_results_filename, 'w') as f:
        json.dump(best_results_out, f, indent=4)
