# Functions for ex2
import matplotlib.pyplot as plt
import numpy as np

# Plots the data points X and y into a new figure
def plotData(X, y):
    pos = y == 1
    neg = y == 0

    plt.plot(X[pos, 0], X[pos, 1], 'k+', markersize = 7, linewidth = 2)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markersize = 7, markerfacecolor = 'y')

# Feature mapping function to polynomial features
def mapFeature(X1, X2):
    degree = 6
    if type(X1) != np.ndarray:
        X1 = np.array([X1])
    s = X1.shape[0]
    out = np.ones([s, 1])
    for i in range(1, degree + 1):  # Note the [1, degree] range
        for j in range(i + 1):      # Note the [0, i] range
            o = np.reshape((X1**(i - j)) * (X2**j), [s, 1])
            out = np.concatenate((out, o), axis = 1)
    return out

# Plots the data points X and y into a new figure with the decision boundary
# defined by theta
def plotDecisionBoundary(theta, X, y):
    plotData(X[:,1:3], y)
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1]) - 2, max(X[:,1]) + 2])

        # Calculate the decision boundary line
        plot_y = np.transpose(np.array([(-1 / theta[2]) * (theta[1] * plot_x + theta[0])]))

        # PLot and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision boundary'])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros([len(u), len(v)])
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.matmul(mapFeature(u[i], v[j]), theta)
        z = np.transpose(z)
        plt.contour(u, v, z, [0, 0], linewidth = 2)

# Compute sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Computes just the gradient. Used for the scipy optimize functions
def gradient(theta, X, y):
    m = len(y)
    n = len(theta)
    y = np.reshape(y, [m, 1])
    theta = np.reshape(theta, [3, 1])  # Make sure theta is the right shape
    h = sigmoid(np.matmul(X, theta))
    return np.sum((h - y) * X, axis = 0) / m

# Computes just the cost. Used for the scipy optimize functions
def costfn(theta, X, y):
    m = len(y)
    y = np.reshape(y, [m, 1])
    theta = np.reshape(theta, [3, 1])  # Make sure theta is the right shape
    h = sigmoid(np.matmul(X, theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m

# Compute cost and gradient for logistic regression
def costFunction(theta, X, y):
    m = len(y)
    y = np.reshape(y, [m, 1])  # make y a column vector

    # Hypothesis function
    h = sigmoid(np.matmul(X, theta))

    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    grad = np.sum((h - y) * X, axis = 0) / m
    
    return [J, grad]

def gradientReg(theta, X, y, lam):
    m = len(y)  # number of training examples
    n = theta.shape
    grad = np.zeros(n)
    y = np.reshape(y, [m, 1])
    theta = np.reshape(theta, [len(theta), 1])
    h = sigmoid(np.matmul(X, theta))
    # grad[0] isn't regularized
    grad[0] = np.sum((h - y) * X[:,0]) / m
    g1 = np.sum((h - y) * X[:,1:], axis = 0)
    g2 = lam * theta[1:] / m
    grad[1:] =  list(np.reshape(g1, [len(theta) - 1, 1]) + g2)
    return grad

def costfnReg(theta, X, y, lam):
    m = len(y)  # number of training examples
    n = theta.shape
    y = np.reshape(y, [m, 1])
    theta = np.reshape(theta, [len(theta), 1])
    h = sigmoid(np.matmul(X, theta))
    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m + lam * np.sum(theta[1:]**2) / (2 * m)
    return J

# Compute cost and gradient for logistic regression with regularization
def costFunctionReg(theta, X, y, lam):
    m = len(y)  # number of training examples
    n = theta.shape
    grad = np.zeros(n)
    y = np.reshape(y, [m, 1])

    h = sigmoid(np.matmul(X, theta))
    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m + lam * np.sum(theta[1:]**2) / (2 * m)

    # grad[0] isn't regularized
    grad[0] = np.sum((h - y) * X[:,0]) / m
    gp = np.sum((h - y) * X[:,1:], axis = 0)
    grad[1:] = np.reshape(gp, [n[0] - 1, 1]) + lam * theta[1:] / m

    return [J, grad]


# Predict whether the label is 0 or 1 using learned logistic regression
# parameters theta
def predict(theta, X):
    m = X.shape[0]  # number of training examples
    p = np.zeros([m, 1])
    p[sigmoid(np.matmul(X, theta)) >= 0.5] = 1
    return p

    