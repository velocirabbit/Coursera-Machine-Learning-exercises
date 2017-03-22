gradientDescent = function(X, y, theta, alpha, numIters) {
  m = length(y)
  J_history = rep(1, numIters)
  for (iter in 1:numIters) {
    theta = theta - alpha * as.vector((t(X) %*% ((X %*% theta) - y)) / m)
    # Calculate cost for this iteration
    J_history[iter] = computeCost(X, y, theta)
  }
  return(list(theta = theta, Jhist = J_history))
}