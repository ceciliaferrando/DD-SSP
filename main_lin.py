import argparse
import dill
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")

from ddssp import *
from utils import *
from baselines import *

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

    res_out = []
    col_out = ['dataset', 'method', 'mse', 'r2', 'experiment_n', 'seed', 'n_limit',
               'param', 'epsilon', 'delta']
    outdir = dataset + "_linreg/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    models_dir = "models_linear"

    ##### Experiment over t trials  ####################################################################################

    for t in range(num_experiments):

        for epsilon in [0.05, 0.1, 0.5, 1.0, 2.0]:

            print(f'epsilon: {epsilon}, t: {t}')

            ##### bounds  ##############################################################################################
            bound_XTX, bound_X = get_bound_XTX(attribute_dict, target, one_hot)
            bound_y = 1 if one_hot else np.abs(np.max(attribute_dict[target]))

            XTX_public = pd.DataFrame(np.dot(X.T, X), index=X.columns, columns=X.columns)
            XTy_public = pd.DataFrame(np.dot(X.T, y), index=X.columns)

            ##### public  ##############################################################################################

            if method == 'public':
                theta_public = public_linreg(X, y)
                theta_public = theta_public.reindex(index=all_columns, fill_value=0)
                mse_public = testLinReg(theta_public, X_test, y_test)
                print(f"public mse {mse_public}")
                res_out.append([dataset, method, mse_public, None, t, seed,
                                n_limit, None, epsilon, delta])

            ##### adassp  ##############################################################################################

            elif method == 'adassp':
                XTy_budget_split = 0.05
                theta_adassp, XTX_dp_reg, XTy_dp = AdaSSP_linear_regression(X, y, epsilon, delta, rho, bound_X, bound_y,
                                                                            bound_XTX, XTy_budget_split)
                theta_adassp = theta_adassp.reindex(index=all_columns, fill_value=0)
                mse_adassp = testLinReg(theta_adassp, X_test, y_test)

                XTX_dp_reg = pd.DataFrame(XTX_dp_reg, index=X.columns, columns=X.columns)
                XTy_dp = pd.DataFrame(XTy_dp, index=X.columns)

                XTX_dp_error, XTy_dp_error = get_SS_mse(XTX_dp_reg, XTX_public,
                                                          XTy_dp, XTy_public)
                print(f"adassp mse {mse_adassp}")
                res_out.append([dataset, method, mse_adassp, None, t, seed,
                                n_limit, None, epsilon, delta])

            ##### DDSSP & AIM Synth  ###################################################################################

            elif method == 'aim':

                file_id = f'{dataset}_epsilon{epsilon}_delta{delta}_nlimit{n_limit}_seed{seed}_t{t}'
                if os.path.exists(f'{models_dir}/aim_model_{file_id}.pkl'):
                    with open(f'{models_dir}/aim_model_{file_id}.pkl', 'rb') as f:
                        aim_model_graph = dill.load(f)
                        mrgs = selectTargetMarginals(pgm_train_df.columns, target, mode=PGMmarginals)
                        workload = {(m, 1.0) for m in mrgs}
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
                ZTZ = get_ZTZ(W_expanded, attribute_dict, all_attributes_expanded, features_to_encode, target, one_hot, scale_y)
                theta_dpqueryss, XTX_dpqueryss, XTy_dpqueryss = dp_query_ss_linreg(ZTZ, all_columns, target)

                mse_dpqueryss = testLinReg(theta_dpqueryss, X_test, y_test)
                print(f"aim ss mse {mse_dpqueryss}")
                res_out.append([dataset, "aim_ss", mse_dpqueryss, None, t, seed,
                                n_limit, None, epsilon, delta])

                # AIM synth
                if one_hot:
                    synth_X = one_hot_encode(synth_X, features_to_encode, attribute_dict)
                    synth_X = normalize_minus1_1(synth_X, attribute_dict, encoded_features)
                    if scale_y:
                        synth_y = normalize_minus1_1(synth_y, attribute_dict, encoded_features)

                zero_std_cols = []
                for col in synth_X.columns:
                    if np.std(synth_X[col]) == 0:
                        print(
                            f"feature {col} is a zero vector! Dropping it at train time, adding corresponding zeros in theta")
                        zero_std_cols.append(col)
                synth_X.drop(columns=zero_std_cols, inplace=True)

                theta_aimsynth = public_linreg(synth_X, synth_y)
                for col in zero_std_cols:
                    theta_aimsynth.loc[col] = 0

                theta_aimsynth = theta_aimsynth.reindex(index=all_columns, fill_value=0)

                for i, col in enumerate(all_columns):
                    assert col == theta_aimsynth.index[i]

                theta_aimsynth = theta_aimsynth.to_numpy()

                mse_aimsynth = testLinReg(theta_aimsynth, X_test, y_test)
                print(f"aim synth mse {mse_aimsynth}")
                res_out.append([dataset, "aim", mse_aimsynth, None, t, seed,
                                n_limit, None, epsilon, delta])

            ##### progressively save  ##################################################################################

            out_df = pd.DataFrame(res_out, columns=col_out)
            filename_out = f'{outdir}{dataset}_{method}_{epsilon}_{t}_{num_experiments}exps_{n_limit}limit_' \
                           f'{seed}seed.csv'
            out_df.to_csv(filename_out)
