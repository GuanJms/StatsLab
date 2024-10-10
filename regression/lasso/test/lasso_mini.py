import numpy as np


def lasso_regression(X, y, lambda_, tol=1e-4, beta=None, fit_intercept=True):
    n, p = X.shape
    y_mean = np.mean(y)
    X_col_mean = np.mean(X, axis=0)

    # Initialize coefficients and intercept
    if fit_intercept:
        beta = np.zeros(p) if beta is None else beta[1:]
        intercept = 0 if beta is None else beta[0]
    else:
        beta = np.zeros(p) if beta is None else beta
        intercept = 0

    XTX = X.T @ X
    YTX = X.T @ y
    X_j_squared = np.array([X[:, j].T @ X[:, j] for j in range(p)])

    # Precompute gamma values for soft-thresholding
    gamma = lambda_ * n / X_j_squared

    def soft_threshold(b, threshold):
        return np.sign(b) * np.maximum(np.abs(b) - threshold, 0)

    # Iterative coordinate descent
    change = np.inf
    while change > tol:
        change = 0

        # Update intercept
        if fit_intercept:
            intercept = y_mean - np.dot(X_col_mean, beta)

        for j in range(p):
            residual = YTX[j] - np.dot(XTX[j, :], beta) + XTX[j, j] * beta[j]
            beta_hat_j = (residual - n * intercept * X_col_mean[j]) / X_j_squared[j]
            new_beta_j = soft_threshold(beta_hat_j, gamma[j])

            # Track largest change in coefficients
            change = max(change, np.abs(beta[j] - new_beta_j))
            beta[j] = new_beta_j

    # Return final coefficients with intercept if requested
    if fit_intercept:
        return np.insert(beta, 0, intercept)
    else:
        return beta
