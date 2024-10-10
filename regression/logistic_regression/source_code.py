import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import fmin_tnc
from numpy import log

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prob(theta, x):
    return sigmoid(np.dot(x, theta))


def objective(theta, x, y):
    p = prob(theta, x)
    return -np.sum(y * log(p) + (1 - y) * log(1 - p))


def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    return np.dot(x.T, sigmoid(np.dot(x, theta)) - y)


def fit(x, y, theta):
    return fmin_tnc(func=objective, x0=theta, fprime=gradient, args=(x, y))[0]

def backtracking(alpha, beta, theta, g, d, x, y):
    t = 1
    while np.isnan(objective(theta + t * d, x, y)) or objective(theta + t * d, x, y) >= objective(theta, x,
                                                                                                  y) + alpha * t * g.T @ d:
        t = beta * t
    return theta + t * d

def run_backtracking(X, y):
    theta = np.zeros(X.shape[1])
    weights = [theta]
    print("Starting object value:", objective(theta, X, y))

    for _ in range(200000):
        g = gradient(theta, X, y)
        d = -g
        theta = backtracking(alpha=0.25, beta=0.5, theta=theta, g=g, d=d, x=X, y=y)
        weights.append(theta)
    print("After running for some time, the objective vlaue is :", objective(theta, X, y))

def newton(x, y, tol=1e-6):
    theta = np.zeros(X.shape[1])
    weights = [theta]
    objective_values = [objective(theta, X, y)]
    print("Starting object value:", objective(theta, X, y))

    while True:
        H = Hessian(theta, x, y)
        g = gradient(theta, x, y)
        theta = theta - np.linalg.inv(H) @ g
        weights.append(theta)
        objective_values.append(objective(theta, X, y))
        if np.linalg.norm(gradient(theta, x, y)) < tol:
            break
    print("After running for some time, the objective vlaue is :", objective(theta, X, y))
    return theta, weights, objective_values

def backtracking_newton(alpha, beta, theta, x, y):
    g = gradient(theta, x, y)
    d = -np.linalg.inv(Hessian(theta, x, y)) @ g
    t = 1
    while np.isnan(objective(theta + t * d, x, y)) or objective(theta + t * d, x, y) >= objective(theta, x, y) + alpha * t * g.T @ d:
        t = beta * t
        print("Backtracking line search t value: ", t)
    return theta + t * d

def Hessian(theta, x, y):
    # Computes the Hessian of the cost function at the point theta
    p = prob(theta, x)
    W = np.diag(p * (1 - p))
    return x.T @ W @ x

def newton_backtracking(x, y, tol=1e-6):
    theta = np.zeros(X.shape[1])
    weights = [theta]
    objective_values = [objective(theta, X, y)]
    print("Starting object value:", objective(theta, X, y))

    while True:
        theta = backtracking_newton(alpha=0.25, beta=0.5, theta=theta, x=x, y=y)
        weights.append(theta)
        objective_values.append(objective(theta, X, y))
        if np.linalg.norm(gradient(theta, x, y)) < tol:
            break
    print("After running, the objective vlaue is :", objective(theta, X, y))
    return theta, weights, objective_values

theta_star_newton_backtracking, weights_newton_backtracking, objective_values_newton_backtracking = newton_backtracking(X, y, tol=1e-6)