## Machine Learning Online Class (Python implementation)
#  Exercise 6 | Support Vector Machines
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as op
from ex6lib import *

if __name__ == '__main__':
## =============== Part 1: Loading and Visualizing Data ================
    print("\nLoading and visualizing data...")

    # Load data from ex6data1
    # Data will be in X and y. mat_dtype so that y is type int instead of uint8
    data = sio.loadmat('ex6data1', mat_dtype = True)
    X = data['X']
    y = data['y']

    # Plot training data
    plotData(X, y)
    plt.show()

    input("Program paused. Press enter to continue.")

## ==================== Part 2: Training Linear SVM ====================
    print("\nTraining linear SVM...")
    
    # Try changing the C value below and see how the decision boundary varies
    C = 100
    model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
    print(model.w)
    print(model.b)
    visualizeBoundaryLinear(X, y, model)
    plt.show()

    input("Program paused. Press enter to continue.")