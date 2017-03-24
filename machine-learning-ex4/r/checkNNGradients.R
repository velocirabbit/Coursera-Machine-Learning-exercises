checkNNGradients = function(lambda = 0) {
  input_layer_size = 3
  hidden_layer_size = 5
  num_labels = 3
  m = 5
  
  # We generate some "random" test data
  theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
  theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
  
  # Reusing debugInitializeWeights to generate X
  X = debugInitializeWeights(m, input_layer_size - 1)
  y = 1 + t((1:m) %% num_labels)
  
  # Unroll parameters
  nn_params = rbind(matrix(theta1, ncol = 1), matrix(theta2, ncol = 1))

  # Use an anonymous function to make cost function shorthand
  costFunc = function(p) nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lambda)
  cg = costFunc(nn_params)
  cost = cg$cost
  grad = cg$grad
  
  numgrad = computeNumericalGradient(costFunc, nn_params)
  
  # Visually examine the two gradient computations. The two columns should be
  # very similar.
  print(cbind(numgrad, grad))
  cat("The above two columns should be very similar.\n")
  cat("  (Left: numerical gradient || Right: analytical gradient)\n")
  
  # Evaluate the norm of the difference between two solution. If you have a
  # correct implementation, then diff should be less than 1e-9
  diff = norm(numgrad - grad) / norm(numgrad + grad)
  
  cat("If your backpropagation implementation is correct, then the relative\n")
  cat("difference will be small (less than 1e-9).\n")
  cat("  Relative difference:", diff, "\n")
}