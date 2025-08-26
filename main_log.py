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

parser = argparse.ArgumentParser(description='Experiment Inputs')
parser.add_argument('--dataset', help='Dataset', type=str, default='ACSPublicCoverage')
parser.add_argument('--method', help='Method to be used', type=str, default='aim',
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


    col_out = ['dataset', 'method', 'auc', 'experiment_n', 'seed', 'n_limit',
               'param', 'epsilon', 'delta']
    outdir = dataset + "_logreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    models_dir = "models_logistic"

    ##### Experiment over t trials #####################################################################################

    for t in range(num_experiments):

        res_out = []

        for epsilon in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:

            print(f'epsilon: {epsilon}, t: {t}')

            ##### bounds ###############################################################################################
            bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, one_hot)
            bound_y = 1 if one_hot else np.abs(np.max(attribute_dict[target]))

            ##### public ###############################################################################################

            if method == 'public':

                theta_public = public_logreg(X, y, all_columns)

                public_auc = testLogReg(theta_public, X_test, y_test)

                res_out.append([dataset, method, public_auc, t, seed, n_limit, None, epsilon, delta])
                print(f"public auc reg: {public_auc}")


            ##### ObjPert ##############################################################################################

            elif method == 'genobjpert':
                theta_genobjpert = objective_perturbation_method(X, y, epsilon, delta, bound_X, bound_y, all_columns)
                genobjpert_auc = testLogReg(theta_genobjpert, X_test, y_test)

                res_out.append([dataset, method, genobjpert_auc, t, seed, n_limit, None, epsilon, delta])
                print(f"genobjpert auc reg: {genobjpert_auc}")


            ##### DDSSP & AIM Synth ####################################################################################

            elif method == 'aim':

                file_id = f'{dataset}_epsilon{epsilon}_delta{delta}_nlimit{n_limit}_seed{seed}_t{t}'
                if os.path.exists(f'{models_dir}/aim_model_{file_id}.pkl'):
                    with open(f'{models_dir}/aim_model_{file_id}.pkl', 'rb') as f:
                        aim_model_graph = dill.load(f)
                        workload = selectTargetMarginals(pgm_train_df.columns, target, mode=PGMmarginals)
                        workload = {(m, 1.0) for m in workload}
                else:
                    # 1) get AIM model and save it

                    initial_cliques = "y_marginals" if aim_y_mrg_opt else None
                    aim_model_graph, workload = get_aim_model(pgm_train_df, domain, target, PGMmarginals, epsilon, delta, model_size,
                                                        max_iters, len(X), initial_cliques)

                    with open(f'{models_dir}/aim_model_{file_id}.pkl', 'wb') as f:
                        dill.dump(aim_model_graph, f)

                    # 2) load AIM model and get marginal tables and synthetic data X_synth, y_synth
                    with open(f'{models_dir}/aim_model_{file_id}.pkl', 'rb') as f:
                        aim_model_graph = dill.load(f)

                aim_ans_wkld = {cl: aim_model_graph.project(cl) for cl, _ in workload}
                W = {key: aim_ans_wkld[key].__dict__['values'] for key in aim_ans_wkld}

                synth = aim_model_graph.synthetic_data(rows=n).df
                synth_X, synth_y = synth.loc[:, synth.columns != target], synth.loc[:, synth.columns == target]

                # DDSSP
                W_expanded = expand_W(W, attribute_dict)
                all_attributes_expanded = all_columns.append(pd.Index([target]))
                ZTZ = get_ZTZ(W_expanded, attribute_dict, all_attributes_expanded, features_to_encode, target, one_hot, True)
                cheb = Chebyshev(-6, 6, 3, phi_logit)   # corresponds
                theta_dpqueryss = dp_query_approx_ss_logreg(ZTZ, all_columns, target, n, cheb, C=1.0)

                theta_dpqueryss = theta_dpqueryss.reindex(index=all_columns, fill_value=0)
                X_test = X_test.reindex(columns=theta_dpqueryss.index, fill_value=0)
                logits = np.dot(X_test, theta_dpqueryss)
                probabilities = 1 / (1 + np.exp(-logits))
                dpqueryss_auc = roc_auc_score(y_test, probabilities)

                res_out.append([dataset, "aim_ss", dpqueryss_auc, t, seed, n_limit, None, epsilon, delta])
                print(f"dpqueryss auc: {dpqueryss_auc}")

                XTX = ZTZ.loc[all_columns, all_columns]
                XTy = ZTZ.loc[all_columns, target]
                b1, b2, b3 = cheb.c  # chebyshev constants
                theta_cf = - b2 / (2 * b3) * np.linalg.solve(XTX, XTy)

                diff = theta_dpqueryss.to_numpy().flatten() - theta_cf
                mse = np.mean(diff ** 2)  # mean squared error across p coefficients
                rmse = np.sqrt(mse)
                rel_l2 = np.linalg.norm(diff) / np.linalg.norm(theta_cf)

                np.save(f'theta_ss_{epsilon}.npy', theta_dpqueryss)
                np.save(f'theta_closedform_{epsilon}.npy', theta_cf)

                logits = np.dot(X_test, theta_cf)
                probabilities = 1 / (1 + np.exp(-logits))
                dpqueryss_cf_auc = roc_auc_score(y_test, probabilities)
                res_out.append([dataset, "aim_ss_closedform", dpqueryss_cf_auc, t, seed, n_limit, None, epsilon,
                                delta])
                print(f"dpqueryss closed form auc: {dpqueryss_cf_auc}")

                # AIM synth
                if one_hot:
                    synth_X = one_hot_encode(synth_X, features_to_encode, attribute_dict)
                    synth_X = normalize_minus1_1(synth_X, attribute_dict, encoded_features)

                synth_y = normalize_minus1_1(synth_y, attribute_dict, encoded_features)

                zero_std_cols = []
                for col in synth_X.columns:
                    if np.std(synth_X[col]) == 0:
                        print(
                            f"feature {col} is a zero vector! Dropping it at train time, adding corresponding zeros in theta")
                        zero_std_cols.append(col)
                synth_X.drop(columns=zero_std_cols, inplace=True)

                theta_aimsynth = public_logreg(synth_X, synth_y, all_columns)
                for col in zero_std_cols:
                    theta_aimsynth.loc[col] = 0

                theta_aimsynth = theta_aimsynth.reindex(index=all_columns, fill_value=0)

                for i, col in enumerate(all_columns):
                    assert col == theta_aimsynth.index[i]

                theta_aimsynth = theta_aimsynth.to_numpy()

                aimsynth_auc = testLogReg(theta_aimsynth, X_test, y_test)
                res_out.append([dataset, "aim", aimsynth_auc, t, seed, n_limit, None, epsilon, delta])
                print("aim synth auc", aimsynth_auc)

        ##### progressively save trials ################################################################################

        out_df = pd.DataFrame(res_out, columns=col_out)
        one_hot_flag_str = '' if not one_hot else 'one-hot_True'
        filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                       f'{seed}seed_{one_hot_flag_str}.csv'
        out_df.to_csv(filename_out)
        print(f"Saved results for trial {t} to {filename_out}")