# Library file for ex6
import matplotlib.pyplot as plt
import numpy as np
import sys
from SVM_Model import SVM_Model

# Plots the data points X and y into a new figure
def plotData(X, y):
    pos = (y == 1).flatten()
    neg = (y == 0).flatten()

    plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth = 1, markersize = 7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markersize = 7, markerfacecolor = 'y')

# Trains an SVM classifier using a simplified version of the SMO algorithm
def svmTrain(X, Y, C, kernelFunction, tol = 1e-3, max_passes = 5):
    # Data parameters
    m, n = X.shape
    
    # Map 0 to -1
    Y = Y.flatten()
    Y[Y == 0] = -1

    # Variables
    alphas = np.zeros([m,])
    b = 0
    E = np.zeros([m,])
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
    if kernelFunction.__name__ == 'linearKernel':
        # Vectorized computation for the linear kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = np.matmul(X, X.T)
    elif kernelFunction.__name__ == 'gaussianKernel':
        # Vectorized RBF (radial basis function, i.e. Gaussian) kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = np.sum(X**2, axis = 1)
        K = X2 + (X2.T - 2*np.matmul(X, X.T))
        K = kernelFunction(1, 0)**K
    else:
        # Pre-compute the kernel matrix
        # This can be slow due to the lack of vectorization
        K = np.zeros([m, m])
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernelFunction(
                    X[i,:].T, X[j,:].T
                )
                K[j, i] = K[i, j]  # the matrix is symmetric
    
    # Train
    print("\nTraining", end = ''); sys.stdout.flush()
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # Calculate Ei = f(x[i]) - y[i]
            # E[i] = b + sum(X[i,:] * np.transpose(np.tile(alphas*Y, 1, n) * X) - Y[i]
            E[i] = b + np.sum(alphas*Y*K[:,i]) - Y[i]
            
            if (Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0):
                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = np.floor(m * np.random.rand()).astype(int)
                while j == i:  # Make sure i != j
                    j = np.floor(m * np.random.rand()).astype(int)

                # Calculate E[j] = f(x[j]) - y[j]
                E[j] = b + np.sum(alphas*Y*K[:,j]) - Y[j]

                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # Compute L and H
                if Y[i] == Y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                
                if L == H:
                    # Continue to the next i
                    continue
                
                # Compute eta
                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    # Continue to next i
                    continue

                # Compute and clip new value for alpha j
                alphas[j] = alphas[j] - (Y[j]*(E[i] - E[j]))/eta

                # Clip
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])


                # Check if change in alpha is significant
                if np.abs(alphas[j] - alpha_j_old) < tol:
                    # Continue to next i, but still replace anyway
                    alphas[j] = alpha_j_old
                    continue

                # Determine value for alpha i
                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])

                # Compute b1 and b2
                b1 = b - E[i] - \
                        Y[i]*(alphas[i] - alpha_i_old) * K[i, j].T - \
                        Y[j]*(alphas[j] - alpha_j_old) * K[i, j].T
                b2 = b - E[j] - \
                        Y[i]*(alphas[i] - alpha_i_old) * K[i, j].T - \
                        Y[j]*(alphas[j] - alpha_j_old) * K[j, j].T

                # Compute b
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2

                num_changed_alphas = num_changed_alphas + 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

        print(".", end = ''); sys.stdout.flush()
        dots += 1
        if dots > 78:
            dots = 0
            print(""); sys.stdout.flush()

    print(" Done! \n")
    print(E)
    # Save the model
    idx = alphas > 0

    model = SVM_Model(kernelFunction, X, Y, alphas, b, idx)

    return model

# Returns a vector of predictions using a trained SVM model (via svmTrain).
def svmPredict(model, X):
    return None  #TODO: finish this

# Plots a non-linear decision boundary learned by the SVM.
def visualizeBoundary(X, y, model, varargin):
    # Plot the training data on top of the boundary
    plotData(X, y)
    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    x2plot = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    #TODO: Finish this

# Plots a linear decision boundary learned by the SVM
def visualizeBoundaryLinear(X, y, model):
    w = model.w
    b = model.b

    xp = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    yp = -(w[0]*xp + b)/w[1]
    plotData(X, y)
    plt.plot(xp, yp, '-b')

# Returns a linear kernel between x1 and x2
def linearKernel(x1, x2):
    # Ensure that x1 and x2 are column vectors
    x1 = np.reshape(x1, [x1.size, 1])
    x2 = np.reshape(x2, [x2.size, 1])

    # Compute the kernel
    sim = np.matmul(x1.T, x2)
    return sim

# Returns a radial basis function kernel between x1 and x2
def gaussianKernel(x1, x2, sigma):
    # Ensure that x1 and x2 are column vectors
    x1 = np.reshape(x1, [x1.size, 1])
    x2 = np.reshape(x2, [x2.size, 1])

    # Compute the kernel
    sim = np.exp(-np.sum((x1 - x2)**2, axis = 0) / (2 * sigma**2))
    return sim

# Reads the fixed vocabulary list in vocab.txt and returns a cell array of the
# words.
def getVocabList():
    with open('vocab.txt', 'r') as f:
        vocabList = [line.split()[1] for line in f]
    return vocabList

# Reads a file and returns its entire contents
def readFile(filename):
    try:
        return open(filename, 'r').read()
    except:
        print('Unable to open %s' % filename)
        return ''
