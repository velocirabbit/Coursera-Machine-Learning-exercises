nnCostFunction = function(nn_params, input_layer_size, hidden_layer_size,
                          num_labels, X, y, lambda) {
  # Unroll nn_params
  paramunroll = hidden_layer_size * (input_layer_size + 1)
  theta1 = matrix(nn_params[1:paramunroll],
                  nrow = hidden_layer_size, ncol = input_layer_size + 1)
  theta2 = matrix(nn_params[paramunroll+1:length(test)],
                  nrow = num_labels, ncol = hidden_layer_size + 1)
  
  m = dim(X)[1]
  
  theta1_grad = matrix(0, nrow = dim(theta1)[1], ncol = dim(theta1)[2])
  theta2_grad = matrix(0, nrow = dim(theta1)[1], ncol = dim(theta1)[2])
  
  ### Initializations
  # Add a column of bias units to each example
  X = cbind(rep(1, m), X)
  
  # Initialize activation value matrix for the hidden layer
  a2 = matrix(0, nrow = m, ncol = hidden_layer_size + 1)
  k = dim(theta2)[1]  # Number of classes
  yBinK = 1:k == y    # For each row of y, compares the sequence 1:k to y
  
  for (i in 1:m) {
    
  }
}