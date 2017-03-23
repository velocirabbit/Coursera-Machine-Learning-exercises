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
cg = lrCostFunction(theta_t, X_t, y_t, lambda_t)

cat("Cost:", cg$cost, "\n")
cat("Expected cost: 2.534819\n")
cat("Gradients:\n", cg$grad, "\n")
cat("Expected gradients:\n 0.146561  -0.548558  0.724722  1.398003\n")

readline(prompt = "Program paused. Press enter to continue.")

## ============ Part 2b: One-vs-All Training ============
cat("Training One-vs-All logistic regression...\n")

lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda)

readline(prompt = "Program paused. Press enter to continue.")

## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)
cat("Training set accuracy:", mean(pred == y) * 100, "%\n")
