## Machine Learning Online Class (R implementation)
# Exercise 5 | Regularized Linear Regression and Bias-Variance
require('rdetools')
require('R.matlab')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = T), source)  # load all fn's
par(mfrow = c(1, 1))

## =========== Part 1: Loading and Visualizing Data =============
cat("\nLoading and visualizing data...\n")

# Load training data from ex5data1
# data will contain: X, y, Xval, yval, Xtest, ytest
data = readMat('ex5data1.mat')
X = data$X; Xval = data$Xval; Xtest = data$Xtest
y = data$y; yval = data$yval; ytest = data$ytest

m = nrow(X)  # number of training examples

# Plot training data
plot(X, y, col = 'red', pch = 'x',
     xlab = 'Change in water level (x)', ylab = 'Water flowing out of the dam (y)')

readline(prompt = "Program paused. Press enter to continue.")

## =========== Part 2: Regularized Linear Regression Cost =============
theta = rep(1, 2)
cg = linearRegCostFunction(cbind(rep(1, m), X), y, theta, 1)
J = cg$cost

cat("Cost at theta = [1; 1]:", J, "\n")
cat("  (this value should be about 303.993192)\n")

readline(prompt = "Program paused. Press enter to continue.")

## =========== Part 3: Regularized Linear Regression Gradient =============
# This part is identical to Part 2, but this time print the gradient.
grad = cg$grad
cat("\nGradient:", grad, "\n")
cat("  (should be [-15.3030; 598.2507])\n")

readline(prompt = "Program paused. Press enter to continue.")

## =========== Part 4: Train Linear Regression =============
cat("\nTraining linear regression and plotting best fit line...\n")

# Train linear regression with lambda = 0
lambda = 0
theta = trainLinearReg(cbind(rep(1, m), X), y, lambda)

# Plot fit over the data
plot(X, y, col = 'red', pch = 'x',
     xlab = 'Change in water level (x)', ylab = 'Water flowing out of the dam (y)')
lines(X, cbind(rep(1, m), X) %*% theta)

readline(prompt = "Program paused. Press enter to continue.")

## =========== Part 5: Learning Curve for Linear Regression =============
cat("\nPlotting learning curve for linear regression...\n")
lambda = 0
errs = learningCurve(cbind(rep(1, m), X), y,
                     cbind(rep(1, nrow(Xval)), Xval), yval, lambda)
error_train = errs$train
error_val = errs$val

plot(1:m, error_train, type = 'l', col = 'blue', xlim = c(0, 13), ylim = c(0, 150),
     ylab = 'Error', xlab = 'Number of training examples',
     main = 'Learning curve for linear regression')
lines(1:m, error_val, col = 'green')
legend(0, 145, c('Train', 'Cross validation'), col = c('blue', 'green'))

cat("# Training examples  |  Training error  |  Cross validation error\n")
for (i in 1:m) {
  cat(paste0("  \t", i, "\t\t   ", error_train[i], "  \t\t", error_val[i], "\n"))
}

readline(prompt = "Program paused. Press enter to continue.")

## =========== Part 6: Feature Mapping for Polynomial Regression =============
cat("\nUsing feature mapping for polynomial regression...\n")

p = 8

# Map X onto polynomial features and normalize
X_poly = polyFeatures(X, p)
normf = featureNormalize(X_poly)  # normalize
X_poly = cbind(rep(1, m), normf$X)  # add a column of 1s
mu = normf$mu
sigma = normf$sigma

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = t((t(X_poly_test) - mu) / sigma)
X_poly_test = cbind(rep(1, nrow(X_poly_test)), X_poly_test)  # add column of 1s

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = t((t(X_poly_val) - mu) / sigma)
X_poly_val = cbind(rep(1, nrow(X_poly_val)), X_poly_val)  # add column of 1s

cat("Normalized training example 1:\n")
cat(" ", X_poly[1,], "\n")

readline(prompt = "Program paused. Press enter to continue.")

## =========== Part 7: Learning Curve for Polynomial Regression =============
cat("\nLearning curve for polynomial regression...\n")
lamb = 1.5
theta = trainLinearReg(X_poly, y, lambda)

# Plot training data and fit
par(mfrow = c(2, 1))
plot(X, y, col = 'red', pch = 'x',
     xlab = 'Change in water level (X)', ylab = 'Water flowing out of the dam (y)',
     main = paste0('Polynomial regression fit (lambda = ', lambda, ')'))
plotFit(min(X), max(X), mu, sigma, theta, p)

errs = learningCurve(X_poly, y, X_poly_val, yval, lambda)
error_train = errs$train
error_val = errs$val
plot(1:m, error_train, type = 'l', xlab = 'Number of training examples',
     xlim = c(0, 13), ylim = c(0, 100), ylab = 'Error',
     main = paste0('Polynomial regression learning curve (lambda =', lambda, ')'))
lines(1:m, error_val)
#legend(0, 95, c('Train', 'Cross validation'))

readline(prompt = "Program paused. Press enter to continue.")
par(mfrow = c(1, 1))

## =========== Part 8: Validation for Selecting Lambda =============
cat("\nValidation for selecting lambda\n")

lambdaErrs = validationCurve(X_poly, y, X_poly_val, yval)
lambda_vec = lambdaErrs$lambdas
error_train = lambdaErrs$train
error_val = lambdaErrs$val

plot(lambda_vec, error_train, type = 'l', xlab = 'lambda', ylab = 'Error')
lines(lambda_vec, error_val)

cat("Lambda  |  Training error  |  Cross validation error\n")
for (i in 1:length(lambda_vec)) {
  cat(paste0("  \t", lambda_vec[i], "\t\t   ", error_train[i], "  \t\t", error_val[i], "\n"))
}


