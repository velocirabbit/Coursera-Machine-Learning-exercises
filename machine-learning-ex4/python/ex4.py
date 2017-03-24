## Machine Learning Online Class (Python implementation) - Exercise 4 Neural Network Learning
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as op
from ex4lib import *

if __name__ == '__main__':
    input_layer_size = 400  # 20x20 input images of digits
    hidden_layer_size = 25  # 25 hidden units
    num_labels = 10         # 10 labels, from 1 to 10 ("0" mapped to label 10)

## =========== Part 1: Loading and Visualizing Data =============
    print("\nLoading and visualizing data...")
    data = sio.loadmat('ex4data1')
    X = data['X']
    y = data['y']
    m, n = X.shape

    # Randomly select 100 data points to display
    rand_indices = np.random.choice(m, size = 100, replace = False)
    sel = X[rand_indices, :]

    displayData(sel)
    plt.show()

    input("Program paused. Press enter to continue.")

## ================ Part 2: Loading Parameters ================
    print("\nLoading saved neural network parameters...")
    params = sio.loadmat('ex4weights')
    theta1 = params['Theta1']
    theta2 = params['Theta2']

    # Going to ignore the parameter unrolling here because it isn't necessary
    # in Python.
    nn_params = np.append(theta1.flatten(), theta2.flatten())
## ================ Part 3: Compute Cost (Feedforward) ================
    print("\nFeedforward using neural network...")

    # Weight regularization parameter (we set this to 0 here)
    lamb = 0

    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                             num_labels, X, y, lamb)
    
    print("Cost at parameters: %.6f" % J)
    print("  (this value should be about 0.287629)")

    input("Program paused. Press enter to continue.")

## =============== Part 4: Implement Regularization ===============
    print("\nChecking cost function (w/ regularization)...")

    # Weight regularization parameter (we now set this to 1 here)
    lamb = 1

    J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                             num_labels, X, y, lamb)
                            
    print("Cost at parameters: %.6f" % J)
    print("  (this value should be about 0.383770)")

    input("Program paused. Press enter to continue.")

## ================ Part 5: Sigmoid Gradient  ================
    print("\nEvaluating sigmoid gradient...")

    g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print("Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]:")
    print("\t", g)
    print("  (these values should be:)")
    print("\t [ 0.196612\t0.235004\t0.250000\t0.235004\t0.196612]")

    input("Program paused. Press enter to continue.")

## ================ Part 6: Initializing Pameters ================
    print("\nInitializing neural network parameters...")

    initialTheta1 = randInitialWeights(input_layer_size, hidden_layer_size)
    initialTheta2 = randInitialWeights(hidden_layer_size, num_labels)

    initial_nn_params = np.append(initialTheta1.flatten(), initialTheta2.flatten())

## =============== Part 7: Implement Backpropagation ===============
    print("\nChecking backpropagation...")

    # Check gradients by running checkNNGradients
    checkNNGradients()

## =============== Part 8: Implement Regularization ===============
    print("\nChecking backpropagation (w/ regularization)")

    # Check gradients by running checkNNGradients
    lamb = 3
    checkNNGradients(lamb)

    # Also output the costFunction debugging values
    debug_J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                             num_labels, X, y, lamb)[0]
            
    print("Cost at (fixed) debugging parameters (w/ lambda = %.3f): %.3f" % (lamb, debug_J))
    print("  (for lambda = 3, this value should be about 0.576051)")

    input("Program paused. Press enter to continue.")

## =================== Part 9: Training NN ===================
    print("\nTraining neural network... ")

    lamb = 1
    costFn = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                      num_labels, X, y, lamb, False)

    # op.minimize() requires/works best when x0 is a single vector of type ndarray
    # if jac = True, costFn is assumed to return the cost as the first return
    # value, and the gradient as the second.
    res = op.minimize(fun = costFn, x0 = initial_nn_params, jac = True,
                        method = 'TNC', options = {'maxiter': 250})
    cost = res.fun
    nn_params = res.x

    # Obtain theta1 and theta2 back from nn_params
    paramunroll = hidden_layer_size * (input_layer_size + 1)
    theta1 = np.reshape(nn_params[:paramunroll], [hidden_layer_size, input_layer_size + 1])
    theta2 = np.reshape(nn_params[paramunroll:], [num_labels, hidden_layer_size + 1])

    input("Program paused. Press enter to continue.")

## ================= Part 10: Visualize Weights =================
    # Visualize what the neural network is learning by displaying the hidden
    # units to see what featuers they are capturing in the data.
    print("\nVisualizing neural network...")
    displayData(theta1[:, 1:])
    plt.show()
    
    input("Program paused. Press enter to continue.")

## ================= Part 11: Implement Predict =================
    # Use the neural network to predict the labels
    pred = predict(theta1, theta2, X)
    ym = np.reshape(y, pred.shape)

    print("\nTraining set accuracy: %.3f%%" % (np.mean(pred == ym) * 100))
