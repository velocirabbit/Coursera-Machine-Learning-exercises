linearRegCostFunction = function(X, y, theta, lambda) {
  y = matrix(y, ncol = 1)  # make sure y is a column vector
  m = length(y)  # number of training examples
  theta = matrix(theta, ncol = 1)
  # Get the linear model values
  # h = theta_0 + theta_1 * x
  h = X %*% theta  # h is of size [m x 1]
  
  # Calculate the cost of this iteration
  J = (sum((h - y)^2) + lambda * sum(theta[-1]^2)) / (2 * m)

  # Get the gradient terms
  #print(length(colSums(sweep(X, MARGIN = 1, h - y, '*'))))
  #print(length(rbind(0, theta[-1])))
  #cat("theta:", length(theta), ", append:", length(append(theta[-1], )), "\n")
  thz = append(theta[-1], 0, 0)
  grad = (colSums(sweep(X, MARGIN = 1, h - y, '*')) + lambda * thz) / m
  grad = matrix(grad, ncol = 1)
  return(list(cost = J, grad = grad))
}