lrcostfn = function(theta, X, y, lambda) {
  m = length(y)  # number of training examples
  n = length(theta)
  h = sigmoid(X %*% theta)
  # Don't include bias unit when calculating the regularization term
  J = -sum(y * log(h) + (1 - y) * log(1 - h)) / m + lambda * sum(theta[2:n]^2) / (2 * m)

  return(J)
}