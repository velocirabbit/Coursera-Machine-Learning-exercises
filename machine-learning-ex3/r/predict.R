predict = function(Theta1, Theta2, X) {
  # Theta1: [hidden_layer_size x n+1]
  # Theta2: [num_labels x hidden_layer_size+1]
  # X: [m x n+1]
  m = dim(X)[1]
  num_labels = dim(Theta2)[1]

  # Add bias unit to each trial X
  X = cbind(rep(1, m), X)
  
  # Calculate hidden layer activations
  a2 = sigmoid(X %*% t(Theta1))  # X %*% t(Theta1): [m x hidden_layer_size]

  # Add bias unit to the hidden layer of each trial
  a2 = cbind(rep(1, m), a2)
  
  # Calculate the output layer values and get the output node index with the
  # largest value - this is the prediction of the whole neural net.
  # p is a vector of the indices of the largest values (what we want)
  vals = sigmoid(a2 %*% t(Theta2))
  p = lapply(1:m, function(i) which.max(vals[i,]))
  return(p)
}