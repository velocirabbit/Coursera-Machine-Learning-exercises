## Machine Learning Online Class (R implementation) - Exercise 2: Logistic Regression
require('rdetools')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = TRUE), source)  # load all fn's

data = read.csv('ex2data1.txt', header = F)
X = data[,1:2]
y = data[,3]

## ==================== Part 1: Plotting ====================
cat("Plotting data with + indicating (y=1) examples and o indicating (y=0) examples.\n")

plotData(X, y, 'Exam 1 score', 'Exam 2 score')
legend(x = 40, y = 100, c('Admitted', 'Not admitted'), pch = c(3, 20), col = c('black', 'orange'))

readline(prompt = "Program paused. Press enter to continue.")

## ============ Part 2: Compute Cost and Gradient ============
mn = dim(X)
m = mn[1]; n = mn[2]

# Add intercept term to x and X_test
X = as.matrix(cbind(rep(1, m), X))

# Initialize fitting parameters
initial_theta = rep(0, n + 1)

# Compute and display initial cost and gradient
cg = costFunction(initial_theta, X, y)
cost = cg$cost
grad = cg$grad

cat(paste0("Cost at initial theta (zeros): ", cost, "\n"))
cat("Gradient at initial theta (zeros):\n")
cat(paste0("  [", grad[1], ", ", grad[2], ", ", grad[3], "]\n"))

readline(prompt = "Program paused. Press enter to continue.")

## ============= Part 3: Optimizing using fminunc  =============
res = optim(initial_theta, fn = costfn, gr = gradient, X = X, y = y, control = list(maxit = 400))
theta = res$par
cost = res$value

cat(paste0("Cost at theta found by optim(): ", cost, "\n"))
cat("Gradient at theta:\n")
cat(paste0("  [", theta[1], ", ", theta[2], ", ", theta[3], "]\n"))

# Plot decision boundary
plotDecisionBoundary(theta, X, y, 'Exam 1 score', 'Exam 2 score')
legend(x = 40, y = 100, c('Admitted', 'Not admitted'), pch = c(3, 20), col = c('black', 'orange'))

readline(prompt = "Program paused. Press enter to continue.")

## ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(c(1, 45, 85) %*% theta) * 100
cat(paste0("For a student with scores 45 and 85, we predict an admission probability of ", prob, "%.\n"))

# Compute accuracy on our training set
p = predict(theta, X)

cat(paste0("Training accuracy: ", mean(p == y) * 100, "%\n"))

