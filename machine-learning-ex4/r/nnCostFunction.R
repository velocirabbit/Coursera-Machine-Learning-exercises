nnCostFunction = function(nn_params, input_layer_size, hidden_layer_size,
                          num_labels, X, y, lambda, printing = F) {
  # Unroll nn_params
  paramunroll = hidden_layer_size * (input_layer_size + 1)
  theta1 = matrix(nn_params[1:paramunroll],
                  nrow = hidden_layer_size, ncol = input_layer_size + 1)
  theta2 = matrix(nn_params[(paramunroll+1):length(nn_params)],
                  nrow = num_labels, ncol = hidden_layer_size + 1)
  
  m = dim(X)[1]
  n = dim(X)[2]
  
  theta1_grad = matrix(0, nrow = dim(theta1)[1], ncol = dim(theta1)[2])
  theta2_grad = matrix(0, nrow = dim(theta2)[1], ncol = dim(theta2)[2])
  
  ### Initializations
  # Add a column of bias units to each example
  X = cbind(rep(1, m), X)
  
  # Initialize activation value matrix for the hidden layer
  a2 = matrix(0, nrow = m, ncol = hidden_layer_size + 1)
  k = dim(theta2)[1]  # Number of classes
  # For each row of y, compares the sequence 1:k to y
  yBinK = t(sapply(y, function(i) 1:k == i))

  if (printing) {
    cat("  |")
  }
  
  for (i in 1:m) {
    ### Feedforward of the input data into the neural net ###
    # For training example i...
    # ... calculate the activation values of the hidden layer
    z2i = X[i,] %*% t(theta1)
    # Add a bias unit of value = 1 to the front
    a2i = cbind(1, sigmoid(z2i))
    a2[i,] = a2i
    
    # ... calculate the output values
    z3i = a2i %*% t(theta2)
    aOuti = sigmoid(z3i)
    
    
    # ... get the cost of this example's output
    yk = yBinK[i,]
    dOi = aOuti - yk  # difference in output and training example
    
    # ... propagate the error back to find the next error cost in using the
    # current theta1 values to calculate the hidden layer's activation values
    d2i_b = dOi %*% theta2  # propagate output layer error back
    d2i = d2i_b[2:length(d2i_b)] * sigmoidGradient(z2i)  # gradient change

    # Accumulate the theta gradient
    theta1_grad = theta1_grad + t(d2i) %*% X[i,]
    theta2_grad = theta2_grad + t(dOi) %*% a2i
    
    if (printing && (i %% (m / 80) == 0)) {
      cat("=")
    }
  }
  
  # Add bias unit to each layer
  bias1 = rep(1, hidden_layer_size)
  biasK = rep(1, k)

  # Get parameter gradients
  theta1_grad = (theta1_grad + lambda * cbind(bias1, theta1[, -1])) / m
  theta2_grad = (theta2_grad + lambda * cbind(biasK, theta2[, -1])) / m
  
  ### Get cost after a single iteration ###
  # Calculate the output values
  aOut = sigmoid(a2 %*% t(theta2))

  # Unregularized cost
  J = -sum(yBinK * log(aOut) + (1 - yBinK) * log(1 - aOut)) / m
  
  # Regularization term
  # Combine thetas for easier sums. Don't regularize the bias unit
  allTheta = cbind(theta1[, -1], t(theta2[, -1]))
  reg = lambda * sum(allTheta^2) / (2 * m)
  
  # Regularize the cost
  J = J + reg
  
  # Unroll gradients
  grad = rbind(matrix(theta1_grad, ncol = 1), matrix(theta2_grad, ncol = 1))
  
  if (printing) {
    cat("| Cost:", J, "\n")
  }
  
  return(list(cost = J, grad = grad))
}