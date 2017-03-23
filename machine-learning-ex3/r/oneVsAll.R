oneVsAll = function(X, y, num_labels, lambda) {
  m = dim(X)[1]
  n = dim(X)[2]
  
  # Add a column of 1s to X
  X = cbind(rep(1, m), X)
  
  all_theta = matrix(0, nrow = num_labels, ncol = n + 1)
  
  # Set initial theta
  initial_theta = rep(0, n + 1)
  
  # Run optim() to obtain the optimal theta for each label
  # For each label, we use y == c as a binary classifier for that label
  for (c in 1:num_labels) {
    res = optim(initial_theta, fn = lrcostfn, gr = lrgrad, method = 'BFGS',
                X = X, y = y == c, lambda = lambda)
    all_theta[c,] = res$par
  }
  return(all_theta)
}