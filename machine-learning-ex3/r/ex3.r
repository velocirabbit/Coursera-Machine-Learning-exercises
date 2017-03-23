## Machine Learning Online Class (R implementation) - Exercise 3 | Part 1: One-vs-all
require('rdetools')
require('R.matlab')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = TRUE), source)  # load all fn's

input_layer_size = 400  # 20x20 input images of digits
num_labels = 10         # 10 labels, from 1 to 10 ("0" is mapped to label 10)

## =========== Part 1: Loading and Visualizing Data =============
# Load training data
cat("Loading and visualizing data...\n")

# Get training data stored in .mat file
data = readMat('ex3data1.mat')
X = data$X
y = data$y

m = dim(X)[1]  # Number of training examples

# Randomly select 100 data points to display
rand_indices = sample(m, 100)
sel = X[rand_indices,]

displayData(sel)

readline(prompt = "Program paused. Press enter to continue.")

## ============ Part 2a: Vectorize Logistic Regression ============
cat("Testing lrCostFunction()\n")

theta_t = c(-2, -1, 1, 2)
X_t = cbind(rep(1, 5), matrix(1:15, nrow = 5) / 10)
y_t = c(1, 0, 1, 0, 1) >= 0.5
lambda_t = 3

