gradient = function(theta, X, y) {
  m = length(y)
  h = sigmoid(X %*% theta)
  return(colSums(sweep(X, MARGIN = 1, h - y, '*')) / m)
}