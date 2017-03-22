## Machine Learning Online Class (R implementation) - Exercise 2: Logistic Regression
require('rdetools')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = TRUE), source)  # load all fn's

data = read.csv('ex2data2.txt', header = F)
X = data[, 1:2]; y = data[, 3]

plotData(X, y, 'Microchip Test 1', 'Microchip Test 2')
legend(x = 0.4, y = 1, c('y = 1', 'y = 0'), pch = c(3, 20), col = c('black', 'orange'))

## =========== Part 1: Regularized Logistic Regression ============
# Add polynomial features
X = mapFeature(X[, 1], X[, 2])

# Initialize fitting parameters
initial_theta = rep(0, dim(X)[2])

# Set regularization parameter lambda to 1
lambda = 1

# Compute and display initial cost and gradient for regularized logistic regression
cg = costFunctionReg(initial_theta, X, y, lambda)
cost = cg$cost

cat(paste0("Cost at initial theta (zeros): ", cost, "\n"))

readline(prompt = "Program paused. Press enter to continue.")

## ============= Part 2: Regularization and Accuracies =============
# Initial fitting parameters
initial_theta = rep(0, dim(X)[2])

# Set regularization parameter lambda to 1
lambda = 1

# Optimize
res = optim(initial_theta, fn = costfnReg, gr = gradientReg, method = 'BFGS', X = X, y = y, lambda = lambda, control = list(maxit = 400))
theta = res$par
cost = res$value

# Plot boundary
plotDecisionBoundary(theta, X, y, 'Microchip Test 1', 'Microchip Test 2')
legend(x = 0.4, y = 1, c('y = 1', 'y = 0'), pch = c(3, 20), col = c('black', 'orange'))

p = predict(theta, X)

cat(paste0("Training accuracy: ", mean(p == y) * 100, "%\n"))
