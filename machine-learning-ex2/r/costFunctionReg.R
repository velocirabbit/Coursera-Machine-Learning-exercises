costFunctionReg = function(theta, X, y, lambda) {
  m = length(y)      # number of training examples
  n = length(theta)  # number of parameters
  grad = rep(0, n)
  
  # Hypothesis function
  h = sigmoid(X %*% theta)
  
  # Cost
  J = -sum(y * log(h) + (1 - y) * log(1 - h)) / m + lambda * sum(theta[2:n]^2) / (2 * m)
  
  # Grad
  grad[1] = colSums((h - y) * X[,1]) / m
  grad[2:n] = (colSums(sweep(X[,2:n], MARGIN = 1, h - y, '*')) + lambda * theta[2:n]) / m
  
  return(list(cost = J, grad = grad))
}