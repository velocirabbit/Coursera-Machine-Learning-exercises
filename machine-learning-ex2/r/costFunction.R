costFunction = function(theta, X, y) {
  m = length(y)
  
  # Hypothesis function for given X and theta
  h = sigmoid(X %*% theta)
  J = -sum(y * log(h) + (1 - y) * log(1 - h)) / m
  grad = colSums(sweep(X, MARGIN = 1, h - y, '*')) / m
  return(list(cost = J, grad = grad))
}