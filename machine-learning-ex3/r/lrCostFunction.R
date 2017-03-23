lrCostFunction = function(theta, X, y, lambda) {
  m = length(y)  # number of training examples
  n = length(theta)
  
  grad = rep(0, n)
  
  h = sigmoid(X %*% theta)
  
  # Don't include bias unit when calculating the regularization term
}