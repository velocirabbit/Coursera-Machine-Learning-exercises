## Machine Learning Online Class (Python implementation) - Exercise 3 | Part 1: One-vs-all
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
from ex3lib import *

if __name__ == '__main__':
    input_layer_size = 400  # 20x20 input images of digits
    num_labels = 10         # 10 labels, from 1 to 10
                            # "0" is mapped to label 10

## =========== Part 1: Loading and Visualizing Data =============
    # Load training data
    print("Loading and visualizing data...")
    data = sio.loadmat('ex3data1')  # training data stored in arrays X and y
    X = data["X"]
    y = data["y"]
    m, n = X.shape  # m = number of training examples, n = parameters

    # Randomly select 100 data points to display
    rand_indices = random.sample(range(m), k = 100)
    sel = X[rand_indices, :]

    displayData(sel)
    plt.show()

    input("Program paused. Press enter to continue.")

## ============ Part 2a: Vectorize Logistic Regression ============
    print("Testing lrCostFunction()")

    # Test case for lrCostFunction
    theta_t = np.array([[-2], [-1], [1], [2]])
    X_t = np.concatenate([np.ones([5, 1]), 
                          np.transpose(np.reshape(range(1, 16), [3, 5]) / 10)],
                        axis = 1)
    y_t = np.array([[1], [0], [1], [0], [1]]) >= 0.5
    lambda_t = 3
    J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

    print("Cost: %.6f" % J)
    print("Expected cost: 2.534819")
    print("Gradients:")
    print(grad)
    print("Expected gradients:\n  0.146561\n  -0.548558\n  0.724722\n  1.398003")

    input("Program paused. Press enter to continue.")

## ============ Part 2b: One-vs-All Training ============
    print("Training One-vs-All logistic regression...")

    lamb = 0.1
    all_theta = oneVsAll(X, y, num_labels, lamb)
    print(all_theta.shape)
    print(all_theta)
## ================ Part 3: Predict for One-Vs-All ================
    pred = predictOneVsAll(all_theta, X)
    print("Training set accuracy: %.3f" % (np.mean(pred == np.reshape(y, [m, 1])) * 100))