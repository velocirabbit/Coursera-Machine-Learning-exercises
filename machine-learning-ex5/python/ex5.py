## Machine Learning Online Class (Python implementation)
# Exercise 5 | Regularized Linear Regression and Bias-Variance
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from ex5lib import *

if __name__ == '__main__':
## =========== Part 1: Loading and Visualizing Data =============
    print("\nLoading and visualizing data...")

    # Load training data from ex5data1
    # data will contain: X, y, Xval, yval, Xtest, ytest
    data = sio.loadmat('ex5data1')
    X = data['X']
    Xval = data['Xval']
    Xtest = data['Xtest']
    y = data['y']
    yval = data['yval']
    ytest = data['ytest']

    # Number of training examples
    m = X.shape[0]

    # Plot the training data
    plt.plot(X, y, 'rx', markersize = 10, linewidth = 1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

    input("Program paused. Press enter to continue.")

## =========== Part 2: Regularized Linear Regression Cost =============
    theta = np.array([[1], [1]])
    J, grad = linearRegCostFunction(np.concatenate([
                        np.ones([m, 1]), X
                    ], axis = 1), y, theta, 1)

    print("Cost at theta = [1; 1]: %.6f" % J)
    print("  (this value should be about 303.993192)")

    input("Program paused. Press enter to continue.")

## =========== Part 3: Regularized Linear Regression Gradient =============
    # This part is identical to Part 2, but this time print the gradient.
    print("\nGradient: ", grad.flatten())
    print("  (should be [-15.3030; 598.2507])")

    input("Program paused. Press enter to continue.")

## =========== Part 4: Train Linear Regression =============
    print("\nTraining linear regression and plotting best fit line...")
    # Train linear regression with lambda = 0
    lamb = 0
    theta = trainLinearReg(np.concatenate([
                    np.ones([m, 1]), X
                ], axis = 1), y, lamb)

    # Plot fit over the data
    plt.plot(X, y, 'rx', markersize = 10, linewidth = 1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, np.matmul(np.concatenate([
        np.ones([m, 1]), X
    ], axis = 1), theta), '--', linewidth = 2)
    plt.show()

    input("Program paused. Press enter to continue.")

## =========== Part 5: Learning Curve for Linear Regression =============
    print("\nPlotting learning curve for linear regression...")

    lamb = 0
    error_train, error_val = learningCurve(np.concatenate([
                                    np.ones([m, 1]), X
                                ], axis = 1), y, np.concatenate([
                                    np.ones([Xval.shape[0], 1]), Xval
                                ], axis = 1), yval, lamb)

    plt.plot(np.arange(1, m + 1), error_train, 'b')
    plt.plot(np.arange(1, m + 1), error_val, 'g')
    plt.title('Learning curve for linear regression')
    plt.axis([0, 13, 0, 150])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend(['Train', 'Cross validation'])

    print("# Training examples  |  Training error  |  Cross validation error")
    for i in range(m):
        print("  \t%d\t\t   %f  \t\t%f" % (i + 1, error_train[i], error_val[i]))

    plt.show()

    input("Program paused. Press enter to continue.")

## =========== Part 6: Feature Mapping for Polynomial Regression =============
    print("\nUsing feature mapping for polynomial regression...")

    p = 8

    # Map X onto polynomial features and normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)  # normalize
    X_poly = np.concatenate([
        np.ones([m, 1]), X_poly                   # add column of 1s
    ], axis = 1)

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = (polyFeatures(Xtest, p) - mu) / sigma
    X_poly_test = np.concatenate([
        np.ones([X_poly_test.shape[0], 1]), X_poly_test  # add column of 1s
    ], axis = 1)

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = (polyFeatures(Xval, p) - mu) / sigma
    X_poly_val = np.concatenate([
        np.ones([X_poly_val.shape[0], 1]), X_poly_val    # add column of 1s
    ], axis = 1)

    print("Normalized training example 1:")
    print("  ", X_poly[0,:])

    input("Program paused. Press enter to continue.")

## =========== Part 7: Learning Curve for Polynomial Regression =============
    lamb = 1.5
    theta = trainLinearReg(X_poly, y, lamb)

    # Plot training data and fit
    plt.subplot(211)
    plt.plot(X, y, 'rx', markersize = 10, linewidth = 1.5)
    plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial regression fit (lambda = %f)' % lamb)

    plt.subplot(212)
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lamb)
    plt.plot(np.arange(1, m + 1), error_train, 'b')
    plt.plot(np.arange(1, m + 1), error_val, 'g')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 100])
    plt.legend(['Train', 'Cross validation'])
    plt.title('Polynomial regression learning curve (lambda = %f)' % lamb)

    print("Polynomial regression (lambda = %f)" % lamb)
    print("# Training examples  |  Training error  |  Cross validation error")
    for i in range(m):
        print("  \t%d\t\t   %f  \t\t%f" % (i + 1, error_train[i], error_val[i]))

    plt.show()

    input("Program paused. Press enter to continue.")

## =========== Part 8: Validation for Selecting Lambda =============
    lambda_vec, error_train, error_val = validationCurve(X_poly, y,
                                                         X_poly_val, yval)

    plt.plot(lambda_vec, error_train, 'b')
    plt.plot(lambda_vec, error_val, 'g')
    plt.legend(['Train', 'Cross validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.axis([0, 10, 0, 20])

    print("Lambda  |  Training error  |  Cross validation error")
    for i in range(len(lambda_vec)):
        print(" %.3f\t\t%f  \t\t%f" % (lambda_vec[i], error_train[i], error_val[i]))

    plt.show()
