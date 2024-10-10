# NetID: sg7993

# This is a lasso regression model using iterative soft-thresholding and coordinate descent.
import numpy as np


def _soft_threshold(b, gamma):
    return np.sign(b) * np.maximum(np.abs(b) - gamma, 0)


def _gamma(n, lmd, x):
    gammas = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        gammas[i] = n * lmd / x[:, i].dot(x[:, i])
    return gammas

def _lasso_coordinate_descent(**params):
    # Compute beta0* using (3.6) and replace beta0 with beta0*
    y_mean = params['y_mean']
    x_col_mean = params['x_col_mean']
    XTX = params['XTX']
    YTX = params['YTX']
    X_j_squared = params['X_j_squared']
    n = params['n']
    gammas = params['gammas']
    beta = params['beta']
    intercept = params['intercept']

    if intercept:
        b0 = y_mean - x_col_mean.dot(beta)
    else:
        b0 = 0
    abs_change = 0

    # For j = 1, 2, ..., p, calculate beta_j* using (3.4) and replace beta_j with beta_j*
    for j in range(len(beta)):
        gamma_j = gammas[j]

        z_j_dot_x_j = YTX[j] - x_col_mean[j] * n * b0 - sum([XTX[j, k] * beta[k] for k in range(len(beta)) if k != j])

        beta_hat_j = z_j_dot_x_j / X_j_squared[j]
        new_beta_j = _soft_threshold(beta_hat_j, gamma_j)
        abs_change = np.abs(beta[j] - new_beta_j) if np.abs(beta[j] - new_beta_j) > abs_change else abs_change
        beta[j] = new_beta_j

    return b0, beta, abs_change


def lasso(x, y, lmd, tol=1e-4, beta=None, intercept=True):
    epsilon = np.inf
    # Get the number of samples and features
    n, p = x.shape

    # Calculate the mean of y and the mean of each column of X
    y_mean = np.mean(y)
    x_col_mean = np.mean(x, axis=0)

    # Calculate gammas
    gammas = _gamma(n, lmd, x)

    if intercept:
        # Initialize the coefficients to zero if not provided
        beta = np.zeros(p) if beta is None else beta[1:]
        b0 = 0 if beta is None else beta[0]
    else:
        beta = np.zeros(p) if beta is None else beta
        b0 = 0

    XTX = x[:].T.dot(x[:])
    YTX = y[:].T.dot(x[:])

    X_j_squared = np.zeros(p)
    for j in range(p):
        X_j_squared[j] = x[:, j].dot(x[:, j])

    params = {
        'y_mean': y_mean,
        'x_col_mean': x_col_mean,
        'XTX': XTX,
        'YTX': YTX,
        'X_j_squared': X_j_squared,
        'n': n,
        'gammas': gammas,
        'beta': beta,
        'intercept': intercept
    }

    while epsilon > tol:
        b0, beta, epsilon = _lasso_coordinate_descent(**params)

    if intercept:
        beta = np.insert(beta, 0, b0)
    return beta
