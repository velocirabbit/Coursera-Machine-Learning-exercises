# Trained SVM model class
import numpy as np

class SVM_Model:
    def __init__(self, kernelFunction, X, y, alphas, b, idx):
        self.kernelFunction = kernelFunction
        self.type = kernelFunction.__name__
        self.X = X[idx,:]
        self.y = y[idx]
        self.b = b
        self.alphas = alphas[idx]
        self.w = np.matmul(np.transpose(alphas*y), X).T
