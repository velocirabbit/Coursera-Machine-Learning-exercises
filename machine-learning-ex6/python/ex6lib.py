# Library file for ex6
import matplotlib.pyplot as plt
import numpy as np

# Plots the data points X and y into a new figure
def plotData(X, y):
    pos = y == 1
    neg = y == 0

    plot(X[pos, 0], X[pos, 1], 'k+', linewidth = 1, markersize = 7)
    plot(X[neg, 0], X[pos, 1], 'ko', markersize = 7, markerfacecolor = 'y')

# Trains an SVM classifier using a simplified version of the SMO algorithm
def svmTrain(X, Y, C, kernelFunction, tol = 1e-3, max_passes = 5):
    # Data parameters
    m, n = X.shape

    # Map 0 to -1
    Y[Y == 0] = -1

    # Variables
    alphas = np.zeros([m, 1])
    b = 0
    E = np.zeros([m, 1])
    passes = 0
    eta = 0
    L = 0
    H = 0

    # Pre-compute the kernel matrix since our dataset is small. In practice,
    # optimized SVM packages that handle large datasets gracefully will _not_
    # do this.
    #
    # We have implemented optimized vectorized version of the kernels here so
    # that the SVM training will run faster
