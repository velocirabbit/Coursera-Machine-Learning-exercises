# Library functions for ex1 and ex1_multi
import matplotlib.pyplot as plt
import numpy as np

# Example function
def warmUpExercise():
    return np.identity(5)

# Compute cost for linear regression
def plotData(x, y):
    plt.figure()
    plt.plot(x, y, 'rx', markersize = 10)
    plt.xlabel('population')
    plt.ylabel('profit')

# Compute cost for linear regresison
def computeCost(X, y, theta):
    m = len(y)  # number of training examples
    return np.sum((np.matmul(X, theta) - y)**2) / (2 * m)

# Performs gradient descent to learn theta
def gradientDescent(X, y, theta, alpha, numIters):
    m = len(y)
    J_history = np.zeros([numIters, 1])
    for iter in range(numIters):
        theta = theta - np.reshape(alpha *
            np.sum((np.matmul(X, theta) - y) * X , axis = 0)
            / m, theta.shape)
        J_history[iter] = computeCost(X, y, theta)
    return [theta, J_history]

# Normalizes the features in X
def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    X_norm[np.isnan(X_norm)] = 0

    return [X_norm, mu, sigma]

# Computes the closed-form solution to linear regression
def normalEqn(X, y):
    theta = np.zeros([np.shape(X)[1], 1])
    Xtrans = np.transpose(X)
    theta = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(Xtrans, X)),
            Xtrans
        ), y
    )
    return theta