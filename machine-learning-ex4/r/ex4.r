## Machine Learning Online Class (R implementation) - Exercise 4 Neural Network Learning
require('rdetools')
require('R.matlab')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = T), source)  # load all fn's

input_layer_size = 400  # 20x20 input images of digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10         # 10 labels, from 1 to 10 ("0" mapped to label 10)

## =========== Part 1: Loading and Visualizing Data =============
cat("\nLoading and visualizing data...\n")

# Load training data
data = readMat('ex4data1.mat')
X = data$X
y = data$y
m = dim(X)[1]  # number of training examples

# Randomly select 100 data points to display
rand_indices = sample(m, 100)
sel = X[rand_indices,]

displayData(sel)

readline(prompt = "Program paused. Press enter to continue.")

## ================ Part 2: Loading Parameters ================
cat("\nLoading saved neural network parameters...\n")
params = readMat('ex4weights.mat')

theta1 = params$Theta1
theta2 = params$Theta2

# Unroll parameters
nn_params = rbind(matrix(theta1, ncol = 1), matrix(theta2, ncol = 1))

## ================ Part 3: Compute Cost (Feedforward) ================
cat("\nFeedforward using neural network...\n")

# Weight regularization parameter (set to 0 here)
lambda = 0
cg = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                    num_labels, X, y, lambda)
