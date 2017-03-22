## Machine Learning Online Class (Python implementation) - Exercise 1: Linear Regression
require('rdetools')
rm(list = ls())  # Clear environment consoles
lapply(list.files(patter = "[.]R$", recursive = TRUE), source)  # load all funcs

## =============== Part 1: Feature Normalization ===============
cat("Loading data...\n")

# Load data
data = read.csv('ex1data2.txt', header = F)
x = data[, 1:2]
y = data[,3]
m = length(y)

# Print out some data points
cat("First 10 examples from the dataset:\n")
for (i in 1:10) {
  cat(paste0("  x = [", x[i, 1], ", ", x[i, 2], "], y = ", y[i], "\n"))
}

readline(prompt = "Program pause. Press enter to continue.")

# Scale features and set them to zero mean
fN = featureNormalize(x)
x = fN[[1]]
mu = as.vector(fN$mu)
sig = as.vector(fN$sigma)

# Add intercept term to X
X = as.matrix(cbind(rep(1, m), x))

## ================= Part 2: Gradient Descent =================

cat("Running gradient descent...\n")

# Choose some alpha value
alpha = 0.01
numIters = 400

# Init theta and run gradient descent
theta = rep(0, 3)
gd = gradientDescent(X, y, theta, alpha, numIters)
theta = gd$theta
J_hist = gd$Jhist

# Plot the convergence graph
plot(1:length(J_hist), J_hist, "l", col = 'black', xlab = 'Number of iterations',
     ylab = 'Cost J')

# Display gradient descent's results
cat(paste0("Theta computed from gradient descent: [", theta[1], ", ", theta[2], 
           ", ", theta[3], "]\n"))


# Estimate the price of a 1650 sq-ft, 3 br house
i = c(1650, 3)
ix = as.vector(cbind(1, t((i - mu) / sig)))
price = ix %*% theta  # ~$289,314.62
cat(paste0("Predicted price of a 1650 sq-ft, 3br house (using gradient descent): $", price))

readline(prompt = "Program pause. Press enter to continue.")

## ================= Part 3: Normal Equations =================
cat("Solving with normal equations...\n")

# Redo using the normal equations
data = read.csv('ex1data2.txt', header = F)
x = data[, 1:2]
y = data[,3]
m = length(y)

# Add intercept term to X
X = as.matrix(cbind(rep(1, m), x))

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display gradient descent's results
cat(paste0("Theta computed from the normal equation: [", theta[1], ", ", theta[2], 
           ", ", theta[3], "]\n"))

# Estimate the price of a 1650 sq-ft, 3 br house
i = c(1, 1650, 3)
price = i %*% theta  # ~$293,081.46
cat(paste0("Predicted price of a 1650 sq-ft, 3br house (using gradient descent): $", price))

