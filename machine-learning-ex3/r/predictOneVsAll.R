predictOneVsAll = function(all_theta, X) {
  m = dim(X)[1]
  num_labels = dim(all_theta)[1]
  
  p = rep(0, m)
  
  # Add column of 1s to the data matrix X
  X = cbind(rep(1, m), X)
  
  vals = sigmoid(X %*% t(all_theta))  # returns [m x num_labels]
  p = lapply(1:m, function(i) which.max(vals[i,]))
  
}