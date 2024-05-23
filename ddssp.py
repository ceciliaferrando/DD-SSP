import math
import sys
sys.path.append('..')

from scipy.special import expit, logsumexp
from scipy.optimize import minimize, fmin_tnc

from private_pgm_local.src.mbi.workload import Workload

from utils import *


def get_aim_model(pgm_train_df, domain, target, marginals_pgm, epsilon, delta, model_size, max_iters, n_samples, initial_cliques):
    pgm_dataset = Dataset(pgm_train_df, domain)
    mrgs = selectTargetMarginals(pgm_train_df.columns, target, mode=marginals_pgm)
    mrgs = {(m, 1.0) for m in mrgs}
    mrgs_wkld = Workload(mrgs)
    y_pairs = [cl[0] for cl in mrgs_wkld if target in cl[0]] if initial_cliques == "y_marginals" else None
    print(f"initial cliques set to {y_pairs}")
    aim_model = aim.AIM(epsilon=epsilon, delta=delta, max_iters=max_iters, max_model_size=model_size)
    aim_model, synth  = aim_model.run(pgm_dataset, mrgs_wkld, n_samples, initial_cliques=y_pairs)
    return aim_model, mrgs_wkld

def get_ZTZ(W, attribute_dict, columns, features_to_encode, target, rescale, scale_y):
    """
        W: [dict] marginal tables in the form {("feature_A", "feature_B"); m_A x m_B np.array of counts, ...}
        attribute_dict: [dict] attribute levels, {"attr_A": list of ordered possible levels for attr_A, ...}
                                - should include target
        columns: [list] list of names of post-encoding attributes
        features_to_encode: [list] list of features requiring 1-hot encoding}
        rescale: [boolean] True if rescaling numerical non binary attributes in [-1, 1], TBC
    """

    # initialize ZTZ as a DataFrame with *named* columns and rows
    base_matrix = np.zeros((len(columns), len(columns)))
    ZTZ = pd.DataFrame(base_matrix, columns=columns, index=columns)

    # loop through attribute pairs

    for a, attr_a in enumerate(columns):
        for b, attr_b in enumerate(columns[a:]):

            # root name of the attributes
            attr_a_orig = attr_a.split("_")[0]
            attr_b_orig = attr_b.split("_")[0]
            # possible level values
            a_values = attribute_dict[attr_a_orig]
            b_values = attribute_dict[attr_b_orig]
            if rescale:
                if attr_a_orig not in features_to_encode and attr_a_orig != target:
                    a_range_min, a_range_max = min(a_values), max(a_values)
                    a_values = [(1 - (-1)) * ((val - a_range_min) / (a_range_max - a_range_min)) - 1 for val in
                                a_values]
                if attr_b_orig not in features_to_encode and attr_b_orig != target:
                    b_range_min, b_range_max = min(b_values), max(b_values)
                    b_values = [(1 - (-1)) * ((val - b_range_min) / (b_range_max - b_range_min)) - 1 for val in
                                b_values]
                if attr_a_orig == target and scale_y == True:
                    a_range_min, a_range_max = min(a_values), max(a_values)
                    a_values = [(1 - (-1)) * ((val - a_range_min) / (a_range_max - a_range_min)) - 1 for val in
                                a_values]
                if attr_b_orig == target and scale_y == True:
                    b_range_min, b_range_max = min(b_values), max(b_values)
                    b_values = [(1 - (-1)) * ((val - b_range_min) / (b_range_max - b_range_min)) - 1 for val in
                                b_values]

            # case 1: a and b are both ordinal
            if attr_a_orig not in features_to_encode and attr_b_orig not in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                for j, j_value in enumerate(a_values):
                    for k, k_value in enumerate(b_values):
                        ZTZ[attr_a][attr_b] += j_value * k_value * mu_ab[j, k]

            # case 2.1: a is ordinal, b is encoded
            elif attr_a_orig not in features_to_encode and attr_b_orig in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                t = int(float(
                    attr_b.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                ZTZ[attr_a][attr_b] = np.sum(np.multiply(mu_ab[:, t], a_values))

            # case 2.2: a is encoded, b is ordinal
            elif attr_a_orig in features_to_encode and attr_b_orig not in features_to_encode:
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                s = int(float(
                    attr_a.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                ZTZ[attr_a][attr_b] = np.sum(np.multiply(mu_ab[s, :], b_values))

            # case 3: a and b are both encoded
            elif attr_a_orig in features_to_encode and attr_b_orig in features_to_encode:
                s = int(float(
                    attr_a.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                t = int(float(
                    attr_b.split("_")[-1]))  # get level number ***** ASSUMES LEVELS CORRESPOND TO THE NAMES ******
                mu_ab = W[(attr_a_orig, attr_b_orig)]
                ZTZ[attr_a][attr_b] = mu_ab[s, t]

    # copy lower tri to upper tri
    ZTZ = ZTZ + ZTZ.T - np.diag(np.diag(ZTZ))

    return ZTZ

# linear regression ####################################################################################################

def dp_query_ss_linreg(ZTZ, all_columns, target):

    d = len(all_columns)

    XTX = ZTZ.loc[all_columns, all_columns]
    XTy = ZTZ.loc[all_columns, target]

    # get estimator
    XTX = XTX + np.eye(d) * 1e-12    # numerical stability
    theta_query_ss = np.linalg.solve(XTX, XTy)
    theta_query_ss = pd.DataFrame(theta_query_ss)
    theta_query_ss = theta_query_ss.set_index(all_columns)

    return theta_query_ss, XTX, XTy

# logistic regression ##################################################################################################

class Chebyshev:
    """
    Chebyshev(a, b, n, func)
    Given a function func, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.
    Method eval(x) yields the approximated function value.
    """

    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                             for k in range(n)]) for j in range(n)]

    def eval(self, x):
        a, b = self.a, self.b
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)  # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:  # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]  # Last step is different

def phi_logit(x):
    return -math.log(1 + math.exp(-x))

def get_dp_approx_ll(theta, yTX, XTXy2, a, b, c, n):
    dp_approx_ll = n * a + np.dot(theta, yTX) * b + np.dot(np.dot(theta, XTXy2), theta) * c
    return dp_approx_ll

class SSApproxLL():

    def __init__(self, ch, yTX, XTXy2, n, penalty, alpha):
        self.n = n
        self.ch = ch
        self.penalty = penalty
        self.alpha = alpha
        self.theta = None
        self.yTX = yTX
        self.XTXy2 = XTXy2

    def fit(self):
        self.optimize()
        return self

    def log_likelihood(self, theta):
        a, b, c = self.ch.c
        term = get_dp_approx_ll(theta, self.yTX, self.XTXy2, a, b, c, self.n)
        term = 1 / self.n * term
        return term

    def optimize(self):

        def l2_penalty(theta):
            return np.sum(theta ** 2)

        x0 = [.0] * len(self.yTX)

        if self.penalty == None:
            res = minimize(lambda theta: -self.log_likelihood(theta),
                           x0,
                           method='L-BFGS-B',
                           options={'maxiter': 10000},
                           tol=0.00001)
            theta_star = res.x
            fun = res.fun

        elif self.penalty == 'l2':
            res = minimize(lambda theta: -self.log_likelihood(theta) + self.alpha * l2_penalty(theta),
                           x0,
                           method='L-BFGS-B',
                           options={'maxiter': 10000},
                           tol=0.00001)
            theta_star = res.x
            fun = res.fun - self.alpha * l2_penalty(theta_star)

        else:
            raise ValueError('Unknown penalty type, choose None or l2')

        self.theta = theta_star

    def predict_proba(self, X):
        z = np.dot(X, self.theta.T)
        y_pred_proba = expit(z)
        return (y_pred_proba)

    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.theta.T)
        y_pred_proba = expit(z)
        y_pred = 2 * ((y_pred_proba >= threshold).astype(int)) - 1
        return (y_pred)

def dp_query_approx_ss_logreg(ZTZ, all_columns, target, n, cheb, C=1.0):

    XTXy2 = ZTZ.loc[all_columns, all_columns]
    XTy = ZTZ.loc[all_columns, target]
    alpha = 1 / (n * C)
    model = SSApproxLL(cheb, XTy, XTXy2, n, penalty=None, alpha=alpha)
    model.fit()
    theta = model.theta
    theta = pd.DataFrame(theta)
    theta.set_index(all_columns, inplace=True)

    return theta


# ddssp utils ##########################################################################################################

def get_SS_mse(XTX_private, XTX_public, XTy_private, XTy_public):
    if not XTX_private.index.equals(XTX_public.index):
        XTX_private, XTX_public = XTX_private.align(XTX_public, axis=0, join='outer', fill_value=0)
        XTX_private, XTX_public = XTX_private.align(XTX_public, axis=1, join='outer', fill_value=0)

    if not XTy_private.index.equals(XTy_public.index):
        XTy_private, XTy_public = XTy_private.align(XTy_public, axis=0, join='outer', fill_value=0)

    return (np.sqrt(mean_squared_error(XTX_private, XTX_public)),
            np.sqrt(mean_squared_error(XTy_private, XTy_public)))

