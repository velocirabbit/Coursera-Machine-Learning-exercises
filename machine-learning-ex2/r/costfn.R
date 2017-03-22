costfn = function(theta, X, y) {
  m = length(y)
  h = sigmoid(X %*% theta)
  return(-sum(y * log(h) + (1 - y) * log(1 - h)) / m)
}