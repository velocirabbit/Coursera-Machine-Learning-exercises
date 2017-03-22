## Machine Learning Online Class (Python implementation) - Exercise 2: Logistic Regression
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from ex2lib import *

if __name__ == '__main__':
    data = np.loadtxt('ex2data1.txt', delimiter = ',')
    X = data[:,0:2]
    y = data[:,2]
## ==================== Part 1: Plotting ====================
    print("Plotting data with + indicating (y=1) examples, and o indicating (y=0) examples.")

    plotData(X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

    input("Program paused. Press enter to continue.")

## ============ Part 2: Compute Cost and Gradient ============
    # Setup the data matrix appropriately, and add ones for the intercept
    [m, n] = np.shape(X)

    # Add intercept term to x and X_test
    X = np.concatenate([np.ones([m, 1]), X], axis = 1)

    # Initialize fitting parameters
    initial_theta = np.zeros([n + 1, 1])

    # Compute and display initial cost and gradient
    [cost, grad] = costFunction(initial_theta, X, y)

    print("Cost at initial theta (zeros): %.3f" % cost)
    print("Gradient at initial theta (zeros):\n\t[%.3f, %.3f, %.3f]" %
            (grad[0], grad[1], grad[2]))

    input("Program paused. Press enter to continue.")

## ============= Part 3: Optimizing using fminunc  =============
    # Run op.minimize() to obtain the optimal theta
    res = op.minimize(fun = costfn, x0 = initial_theta, args = (X, y),
                method = 'TNC', jac = gradient, options = {'maxiter': 400})
    cost = res.fun
    theta = res.x

    # Print theta to screen
    print("Cost at theta found by op.minimize: %.3f" % cost)
    print("  theta:\n\t[%.3f, %.3f, %.3f]" % (theta[0], theta[1], theta[2]))

    plotDecisionBoundary(theta, X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

    input("Program paused. Press enter to continue.")

## ============== Part 4: Predict and Accuracies ==============
    prob = sigmoid(np.matmul(np.array([1, 45, 85]), theta))
    print("For a student with scores 45 and 85, we predict an admission probability of %.3f" % prob)

    # Compute accuracy on our training set
    p = predict(theta, X)

    print("Training accuracy: %.3f" % (np.mean(p == np.reshape(y, [m, 1])) * 100))
    
