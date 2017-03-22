plotDecisionBoundary = function(theta, X, y, xlabel, ylabel) {
  plotData(X[, 2:3], y, xlabel, ylabel)
  
  if (dim(X)[2] <= 3) {
    # Only need 2 points to define a line, so chose two endpoints
    plot_x = c(min(X[,2]) - 2, max(X[,2]) + 2)
    
    # Calculate the decision boundary line
    plot_y = (-1 / theta[3]) * (theta[2] * plot_x + theta[1])
    
    # Plot decision boundary line
    lines(plot_x, plot_y)
  }
  else {
    u = seq(-1, 1.5, length = 50)
    v = seq(-1, 1.5, length = 50)
    
    z = matrix(0, nrow = length(u), ncol = length(v))
    
    # Evaluate z = theta * x over the grid
    for (i in 1:length(u))
      for (j in 1:length(v))
        z[i, j] = mapFeature(u[i], v[j]) %*% theta
    z = t(z)
    
    # Plot z = 0
    contour(u, v, z, add = T)
  }
}