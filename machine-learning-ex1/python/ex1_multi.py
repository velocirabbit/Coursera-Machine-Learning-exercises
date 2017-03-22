## Machine Learning Online Class (Python implementation) - Exercise 1: Linear Regression
import matplotlib.pyplot as plt
import numpy as np
from ex1lib import *

if __name__ == '__main__':
## =============== Part 1: Feature Normalization ===============
    print("Loading data...")

    # Load data
    data = np.loadtxt('ex1data2.txt', delimiter = ',')
    m = len(data)  # number of training examples
    X = np.reshape(data[:,0:2], [m, 2])
    y = np.reshape(data[:,2], [m, 1])

    # Print out some data points
    print("First 10 examples from the dataset:")
    for i in range(10):
        print("  x = [%.3f, %.3f], y = %.3f" % (X[i, 0], X[i, 1], y[i]))

    input("Program paused. Press enter to continue.")

    # Scale features and set them to zero mean
    print("Normalizing features...")

    X, mu, sigma = featureNormalize(X)

    # Add intercept term to X
    X = np.concatenate((np.ones([m, 1]), X), axis = 1)

## ================= Part 2: Gradient Descent =================
    print("Running gradient descent...")

    # Choose some alpha value
    alpha = 0.01
    numIters = 400

    # Init theta and run gradient descent
    theta = np.zeros([3, 1])
    theta, J_history = gradientDescent(X, y, theta, alpha, numIters)

    # Plot the convergence graph
    jiters = [j + 1 for j in range(np.size(J_history))]
    plt.plot(jiters, J_history, '-b', linewidth = 2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display gradient descent's result
    print("Theta computed from gradient descent:")
    print(" [%.3f, %.3f, %.3f]" % (theta[0], theta[1], theta[2]))

    # Estimate the price of a 1650 sq-ft, 3 br house
    i = [1650, 3];
    ix = np.concatenate(([1], (i - mu) / sigma))
    price = np.matmul(ix, theta)  # price = $289,314.62
    print("Predicted price of a 1650 sq-ft, 3 br house")
    print("(using gradient descent):\n  $%.2f" % price)

    input("Program paused. Press enter to continue.")

## ================= Part 3: Normal Equations =================
    print("Solving with normal equations...")

    # Redo using the normal equation
    data = np.loadtxt('ex1data2.txt', delimiter = ',')
    m = len(data)  # number of training examples
    X = np.reshape(data[:,0:2], [m, 2])
    y = np.reshape(data[:,2], [m, 1])

    # Add intercept term to X
    X = np.concatenate((np.ones([m, 1]), X), axis = 1)

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # Display normal equation's results
    print("Theta computed from the normal equations:")
    print(" [%.3f, %.3f, %.3f]" % (theta[0], theta[1], theta[2]))

    # Estimate the price of a 1650 sq-ft, 3 br house
    ix = [1, 1650, 3]
    price = np.matmul(ix, theta)  # price = $293,081.46
    print("Predicted price of a 1650 sq-ft, 3 br house")
    print("(using gradient descent):\n  $%.2f" % price)
