## Machine Learning Online Class (Python implementation) - Exercise 3 | Part 2: Neural Networks
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
import scipy.optimize as op
from ex3lib import *

if __name__ == '__main__':
    input_layer_size = 400  # 20x20 input images of digits
    hidden_layer_size = 25  # 25 hidden units
    num_labels = 10         # 10 labels, from 1 to 10 ("0" mapped to label 10)

## =========== Part 1: Loading and Visualizing Data =============
    print("Loading and visualizing data...")
    data = sio.loadmat('ex3data1')
    X = data["X"]
    y = data["y"]
    m, n = X.shape  # m = number of training examples, n = parameters

     # Randomly select 100 data points to display
    rand_indices = random.sample(range(m), k = 100)
    sel = X[rand_indices, :]

    displayData(sel)
    plt.show()

    input("Program paused. Press enter to continue.")

## ================ Part 2: Loading Pameters ================
    print("Loading saved neural network parameters...")
    nnparams = sio.loadmat('ex3weights')
    Theta1 = nnparams['Theta1']
    Theta2 = nnparams['Theta2']

## ================= Part 3: Implement Predict =================
    pred = predict(Theta1, Theta2, X)
    print("Training set accuracy: %.3f%%" % (np.mean(pred == y) * 100))
    input("Program paused. Press enter to continue.")

    # To give an idea of the network's output, you can also run through the
    # examples one at a time to see what it is predicting.
    
    # Randomly permute examples
    rp = random.sample(range(m), k = m)

    for i in range(m):
        # Display
        print("Displaying example image")
        displayData(X[rp[i], :])
        plt.show()

        pred = predict(Theta1, Theta2, X[rp[i], :])
        print("Neural network prediction: %d (digit %d)" % (pred, y[rp[i]]))

        s = input("Paused - press enter to continue, q to exit.")
        if s == 'q':
            break