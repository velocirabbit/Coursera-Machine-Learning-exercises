# Library file for ex5
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

# Compute the cost and gradient for regularized linear regresion with multiple
# variables.
def linearRegCostFunction(X, y, theta, lamb):
    m = np.alen(y)  # number of training examples
    theta = np.reshape(theta, [np.alen(theta), 1])

    # Get the linear model values
    # h = theta_0 + theta_1 * x
    h = np.matmul(X, theta)  # h is of size [m x 1]

    # Calculate the cost of this iteration
    J = (np.sum((h - y)**2) + lamb * np.sum(theta[1:]**2)) / (2 * m)

    # Get the gradient terms
    grad = (np.sum((h - y) * X, axis = 0) + lamb * np.append(0, theta[1:])) / m

    return [J, grad]

# Trains linear regression given a dataset (X, y) and a regularization
# parameter lambda.
def trainLinearReg(X, y, lamb):
    # Initialize theta
    initial_theta = np.zeros([X.shape[1], 1])

    # Create lambda function for the cost function to be minimized
    costFn = lambda t: linearRegCostFunction(X, y, t, lamb)

    # Minimize using op.minimize()
    # If jac = True, costFn is assumed to return the cost as the first return
    # value, and the gradient as the second.
    res = op.minimize(fun = costFn, x0 = initial_theta, jac = True,
                        method = 'TNC', options = {'maxiter': 200})
    return res.x  # trained theta values

# Generates the training and cross validation set errors needed to plot a
# learning curve
def learningCurve(X, y, Xval, yval, lamb):
    m = X.shape[0]

    # Pre-initialize vectors to store error data
    error_train = np.zeros([m, 1])
    error_val = np.zeros([m, 1])

    for i in range(m):
        iX = X[1:i, :]
        iY = y[1:i, :]

        # Get training error
        theta = trainLinearReg(iX, iY, lamb)
        error_train[i], __ = linearRegCostFunction(iX, iY, theta, 0)

        # Get cross validation error. Use the theta trained over the training
        # set
        error_val[i], __ = linearRegCostFunction(Xval, yval, theta, 0)

    return [error_train, error_val]

# Generate the training and validation errors needed to plot a validation curve
# that we can use to select lambda
def validationCurve(X, y, Xval, yval):
    # Selected values of lambda
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    numL = len(lambda_vec)

    error_train = np.zeros([numL, 1])
    error_val = np.zeros([numL, 1])

    for i in range(numL):
        lamb = lambda_vec[i]

        # Get theta parameters using this lambda
        theta = trainLinearReg(X, y, lamb)

        # Get training error. Note: don't find error of regularization term
        # (use lambda = 0)
        error_train[i], __ = linearRegCostFunction(X, y, theta, 0)

        # Get cross validation errors. Note: don't find error of regularization
        # term (use lambda = 0)
        error_val[i], __ = linearRegCostFunction(Xval, yval, theta, 0)

    return [lambda_vec, error_train, error_val]

# Maps X (a 1D vector) into the p-th power
def polyFeatures(X, p):
    X_poly = np.zeros([X.size, p])
    for i in range(p):
        X_poly[:, i] = np.transpose(X**(i + 1))
    return X_poly

# Normalizes the features in X
def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis = 0)
    X_norm = X_norm / sigma
    return [X_norm, mu, sigma]

# Plots a learned polynomial regression fit over an existing figure
def plotFit(min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get an
    # idea of how the fit will vary outside the range of the data points
    x = np.transpose(np.arange(min_x - 15, max_x + 25, 0.05))

    # Map the X values
    X_poly = (polyFeatures(x, p) - mu) / sigma

    # Add a column of 1s
    X_poly = np.concatenate([
        np.ones([x.shape[0], 1]), X_poly
    ], axis = 1)

    # Plot
    plt.plot(x, np.matmul(X_poly, theta), '--', linewidth = 2)
