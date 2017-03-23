lrgrad = function(theta, X, y, lambda) {
  m = length(y)  # number of training examples
  n = length(theta)
  grad = rep(0, n)
  h = sigmoid(X %*% theta)
  # grad[1] isn't regularized
  grad[1] = sum((h - y) * X[,1]) / m
  grad[2:n] = (colSums(sweep(X[,2:n], MARGIN = 1, h - y, '*')) + lambda * theta[2:n]) / m
  
  return(grad)
}