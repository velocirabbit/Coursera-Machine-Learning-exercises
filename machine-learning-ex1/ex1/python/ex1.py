## Machine Learning Online Class (Python implementation) - Exercise 1: Linear Regression
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
import matplotlib.pyplot as plt
import numpy as np
from ex1lib import *

if __name__ == '__main__':
## =============== Part 1: Basic Function ===============
    print("Running warmUpExercise...")

    print("5x5 Identity Matrix:")
    print(warmUpExercise())

    input("Program paused. Press enter to continue.\n")

## ================== Part 2: Plotting ==================
    print("Plotting data...")

    data = np.loadtxt('ex1data1.txt', delimiter = ',')
    m = len(data)  # number of training examples
    x = np.reshape(data[:,0], [m, 1])
    y = np.reshape(data[:,1], [m, 1])

    # Plot data
    plotData(x, y)
    plt.show()

    input("Program paused. Press enter to continue.\n")

## ============== Part 3: Gradient Descent ==============
    print("Running gradient descent...\n")

    # Add a column of ones to x
    X = np.concatenate((np.ones([m, 1]), x), axis = 1)
    theta = np.zeros([2, 1])  # Initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # Compute and display initial cost
    J = computeCost(X, y, theta)
    print("Initial cost: %.3f" % J)  # Should be ~32.073

    # Run gradient descent
    theta, _ = gradientDescent(X, y, theta, alpha, iterations)

    # Print theta to screen
    print("Theta found by gradient descent: [%.3f, %.3f]" % (theta[0], theta[1]))

    # Plot the linear fit
    plotData(x, y)
    plt.plot(X[:,1], np.matmul(X, theta))
    plt.legend(['Training data', 'Linear regression'])
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.matmul([1, 3.5], theta)
    predict2 = np.matmul([1, 7], theta)
    print("For population = 35,000, we predict a profit of $%.2f" % (predict1 * 10000))
    print("For population = 70,000, we predict a profit of $%.2f" % (predict2 * 10000))

    input("Program pause. Press enter to continue.")

## ======= Part 4: Visualizing J(theta_0, theta_1) =======
    print("Visualizing J(theta_0, theta_1)...")
    
    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # Initialize J_vals to a matrix of 0's
    J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i, j] = computeCost(X, y, t)
    
    J_vals = np.transpose(J_vals)
    plt.contour(theta0_vals, theta1_vals, J_vals,
        levels = np.logspace(-2, 3, 20))
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.plot(theta[0], theta[1], 'rx', markersize = 10, linewidth = 2)
    plt.show()
