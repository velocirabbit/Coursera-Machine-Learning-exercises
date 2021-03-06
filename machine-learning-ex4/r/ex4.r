## Machine Learning Online Class (R implementation) - Exercise 4 Neural Network Learning
require('rdetools')
require('R.matlab')
rm(list = ls())  # Clear environment variables
lapply(list.files(pattern = "[.]R$", recursive = T), source)  # load all fn's

input_layer_size <- 400  # 20x20 input images of digits
hidden_layer_size <- 25  # 25 hidden units
num_labels <- 10         # 10 labels, from 1 to 10 ("0" mapped to label 10)

## =========== Part 1: Loading and Visualizing Data =============
cat("\nLoading and visualizing data...\n")

# Load training data
data <- readMat('ex4data1.mat')
X <- data$X
y <- data$y
m <- nrow(X)  # number of training examples

# Randomly select 100 data points to display
rand_indices <- sample(m, 100)
sel <- X[rand_indices,]

displayData(sel)

readline(prompt = "Program paused. Press enter to continue.")

## ================ Part 2: Loading Parameters ================
cat("\nLoading saved neural network parameters...\n")
params <- readMat('ex4weights.mat')

theta1 <- params$Theta1
theta2 <- params$Theta2

# Unroll parameters
nn_params <- rbind(matrix(theta1, ncol = 1), matrix(theta2, ncol = 1))

## ================ Part 3: Compute Cost (Feedforward) ================
cat("\nFeedforward using neural network...\n")

# Weight regularization parameter (set to 0 here)
lambda <- 0
cg <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                    num_labels, X, y, lambda)
cost <- cg$cost
grad <- cg$grad

cat("Cost at parameters:", cost, "\n")
cat("  (this value should be about 0.287629)\n")

readline(prompt = "Program paused. Press enter to continue.")

## =============== Part 4: Implement Regularization ===============
cat("\nChecking cost function (w/ regularization)...\n")

# Weight regularization parameter (now we set to 1 here)
lambda <- 1

cg <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                    num_labels, X, y, lambda)
cost <- cg$cost
grad <- cg$grad

cat("Cost at parameters:", cost, "\n")
cat("  (this value should be about 0.383770)\n")

readline(prompt = "Program paused. Press enter to continue.")

## ================ Part 5: Sigmoid Gradient  ================
cat("\nEvaluating sigmoid gradient...\n")

g <- sigmoidGradient(c(-1, -0.5, 0, 0.5, 1))
cat("Sigmoid gradient evaluated at c(-1, -0.5, 0, 0.5, 1):\n")
cat(" ", g, "\n")
cat("(these values should be:)\n")
cat("  0.196612 0.235004 0.25 0.235004 0.196612")

readline(prompt = "Program paused. Press enter to continue.")

## ================ Part 6: Initializing Pameters ================
cat("\nInitializing neural network parameters...\n")

initialTheta1 <- randInitializeWeights(input_layer_size, hidden_layer_size)
initialTheta2 <- randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params <- rbind(matrix(initialTheta1, ncol = 1),
                          matrix(initialTheta2, ncol = 1))

## =============== Part 7: Implement Backpropagation ===============
cat("\nChecking backpropagation...\n")
checkNNGradients()

readline(prompt = "Program paused. Press enter to continue.")

## =============== Part 8: Implement Regularization ===============
cat("\nChecking backpropagation (w/ regularization)...\n")

# Check gradients by running checkNNGradients
lambda <- 3
checkNNGradients(lambda)

# Also output the costFunction debugging values
debug_cg <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                          num_labels, X, y, lambda)$cost
cat("Cost at (fixed) debugging parameters (w/ lambda =", lambda, "):", debug_cg, "\n")
cat("  (for lambda = 3, this value should be about 0.576051)\n")

readline(prompt = "Program paused. Press enter to continue.")

## =================== Part 9: Training NN ===================
cat("\nTraining neural network...\n")

lambda <- 1

# Optimize using fmincg:
cat("\n  Training via fmincg...\n")
costFn = function(p) nnCostFunction(p, input_layer_size, hidden_layer_size,
                                    num_labels, X, y, lambda, F)
options = c(MaxIter = 250)
fcg = fmincg(costFn, initial_nn_params, options)
nn_params_fmincg = fcg$params
cost_fmincg = fcg$cost

# Optimize using optim()
cat("\n  Training via optim...\n")
getCostGrad = function(p, getNew) {
  # We'll save the output from nnCostFunction to global variable getCG, then use
  # getNew to denote whether or not we want to run nnCostFunction again. This
  # will only happen when we want to get a new cost, at which point we'll get a
  # new parameter gradient, too.
  if (getNew) {
    assign("getCG", nnCostFunction(p, input_layer_size, hidden_layer_size,
                                   num_labels, X, y, lambda, T),
           envir = .GlobalEnv)
    return(getCG$cost)
  } else {
    return(getCG$grad)
  }
}

getCost <- function(p) getCostGrad(p, T)
getGrad <- function(p) getCostGrad(p, F)

# optim() doesn't have a TNC optimization method
res <- optim(initial_nn_params, fn = getCost, gr = getGrad,
             method = 'BFGS', control = list(maxit = 250))
nn_params_optim <- res$par
cost_optim <- res$value

cat("\nCost via fmincg:", cost_fmincg[length(cost_fmincg)])
cat("\nCost via optim: ", cost_optim, "\n")

if (cost_fmincg[length(cost_fmincg)] < cost_optim) {
  cost = cost_fmincg[length(cost_fmincg)]
  nn_params = nn_params_fmincg
} else {
  cost = cost_optim
  nn_params = nn_params_optim
}

# Obtain theta1 and theta2 back from nn_params
paramunroll <- hidden_layer_size * (input_layer_size + 1)
theta1 <- matrix(nn_params[1:paramunroll],
                nrow = hidden_layer_size, ncol = input_layer_size + 1)
theta2 <- matrix(nn_params[(paramunroll + 1):length(nn_params)],
                nrow = num_labels, ncol = hidden_layer_size + 1)

readline(prompt = "Program paused. Press enter to continue.")

## ================= Part 10: Visualize Weights =================
cat("\nVisualizing neural network...\n")

displayData(theta1[,-1])

readline(prompt = "Program paused. Press enter to continue.")

## ================= Part 11: Implement Predict =================
pred <- predict(theta1, theta2, X)

cat("\nTraining set accuracy:", mean(pred == y) * 100, "%\n")
