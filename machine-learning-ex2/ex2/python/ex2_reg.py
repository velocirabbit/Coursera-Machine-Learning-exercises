## Machine Learning Online Class (Python implementation) - Exercise 2: Logistic Regression
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from ex2lib import *

if __name__ == '__main__':
    data = np.loadtxt('ex2data2.txt', delimiter = ',')
    X = data[:,0:2]
    y = data[:,2]

    plotData(X, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.show()

## =========== Part 1: Regularized Logistic Regression ============
    # Add polynomial features
    # mapFeature() also adds a column of ones for us, so the interation term
    # is handled
    X = mapFeature(X[:,0], X[:,1])

    # Initialize fitting parameters
    initial_theta = np.zeros([X.shape[1], 1])

    # Set regularization parameter lambda to 1
    lam = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    [cost, grad] = costFunctionReg(initial_theta, X, y, lam)

    print("Cost at initial theta (zeros): %.3f" % cost)
    
    input("Program paused. Press enter to continue.")

## ============= Part 2: Regularization and Accuracies =============
    # Initial fitting parameters
    initial_theta = np.zeros([X.shape[1], 1])

    # Set regularization parameter lambda to 1
    lam = 1

    # Run op.minimize() to obtain the optimal theta
    res = op.minimize(fun = costfnReg, x0 = initial_theta, args = (X, y, lam),
                method = 'TNC', jac = gradientReg, options = {'maxiter': 400})
    cost = res.fun
    theta = res.x

    # Plot boundary
    plotDecisionBoundary(theta, X, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()

    input("Program paused. Press enter to continue.")

## ============== Part 4: Predict and Accuracies ==============
    # Compute accuracy on our training set
    prob = sigmoid(np.matmul(np.array([1, 45, 85]), theta))
    p = predict(theta, X)

    print("Training accuracy: %.3f" % (np.mean(p == np.reshape(y, [m, 1])) * 100))
    

