plotFit = function(min_x, max_x, mu, sigma, theta, p) {
  # We plot a range slightly bigger than the min and max values to get an idea
  # of how the fit will vary outside the range of the data points
  x = as.matrix(seq(min_x - 15, max_x + 25, 0.05))
  X_poly = t((t(polyFeatures(x, p)) - mu) / sigma)
  
  # Add a column of ones
  X_poly = cbind(rep(1, nrow(x)), X_poly)
  
  # Plot to existing plot
  lines(x, X_poly %*% theta)
}