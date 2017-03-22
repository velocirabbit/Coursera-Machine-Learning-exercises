## Machine Learning Online Class (R implementation) - Exercise 1: Linear Regression
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
require('rdetools')
rm(list = ls())  # Clear environment consoles
lapply(list.files(patter = "[.]R$", recursive = TRUE), source)  # load all funcs

## =============== Part 1: Basic Function ===============
cat("Running warmUpExercise...\n")
cat("5x5 Identity Matrix:\n")
print(warmUpExercise())

readline(prompt = "Program paused. Press enter to continue.")

## ================== Part 2: Plotting ==================
cat("Plotting data...\n")
data = read.csv('ex1data1.txt', header = F)
x = data[,1]
y = data[,2]
m = length(y)

plotData(x, y)

readline(prompt = "Program pause. Press enter to continue.")

## ============== Part 3: Gradient Descent ==============
cat("Running gradient descent...\n")

X = cbind(rep(1, m), x)  # Add a column of ones to x
theta = rep(0, 2)        # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# Compute and display initial cost
J = computeCost(X, y, theta)
cat("Initial cost: ", J, "\n")  # Should be ~32.073

# Run gradient descent
gd = gradientDescent(X, y, theta, alpha, iterations)
theta = gd$theta

# Print theta to screen
cat(paste0("Theta found by gradient descent: [", gd$theta[1], ", ", gd$theta[2], "]\n"))

# Plot the linear fit
lines(X[,2], X %*% theta)
legend(x = 5, y = 25, legend = c("Training data", "Linear regression"),
       pch = c('x', 'l'), col = c('red', 'black'))

# Predict values for population sizes of 35,000 and 70,000
ps1 = c(1, 3.5); ps2 = c(1, 7)
predict1 = ps1 %*% theta
predict2 = ps2 %*% theta
cat(paste0("For population = 35,000, we predict a profit of $", 10000*predict1, "\n"))
cat(paste0("For population = 70,000, we predict a profit of $", 10000*predict2, "\n"))

readline(prompt = "Program pause. Press enter to continue.")

## ======= Part 4: Visualizing J(theta_0, theta_1) =======
cat("Visualizing J(theta_0, theta_1)...\n")

# Grid over which we will calculate J
theta0_vals = seq(-10, 10, length = 100)
theta1_vals = seq(-1, 4, length = 100)

# Initialize J_vals to a matrix of 0's
J_vals = matrix(0, length(theta0_vals), length(theta1_vals))

# Fill out J_vals
for (i in 1:length(theta0_vals)) {
  for (j in 1:length(theta1_vals)) {
    t = c(theta0_vals[i], theta1_vals[j])
    J_vals[i, j] = computeCost(X, y, t)
  }
}
J_vals = t(J_vals)
contour(x = theta0_vals, y = theta1_vals, z = J_vals,
        levels = logspace(-2, 3, n = 20), xlab = 'theta_0', ylab = 'theta_1')
points(theta[1], theta[2], col = 'red', pch = 'x')
