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
    print('Model) W = %s, b = %s' % (str(model.w), str(model.b)))
    visualizeBoundaryLinear(X, y, model)
    plt.show()

    input("Program paused. Press enter to continue.")

## ================ Part 3: Implementing Gaussian Kernel ================
# Implement the Gaussian kernel to use with the SVM.
print("\nEvaluating the Gaussian Kernel...")

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f:' + \
      '\n\t%f\n(for sigma = 2, this value should be about 0.324652)' % (
          sigma, sim
      )
)

input("Program paused. Press enter to continue.")

## =================== Part 4: Visualizing Dataset 2 ===================