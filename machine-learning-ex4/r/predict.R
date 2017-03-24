predict = function(theta1, theta2, X) {
  m = dim(X)[1]
  num_labels = dim(theta2)[1]
  
  h1 = sigmoid(cbind(rep(1, m), X) %*% t(theta1))
  h2 = sigmoid(cbind(rep(1, m), h1) %*% t(theta2))
  p = lapply(1:m, function(i) which.max(h2[i,]))
  return(p)
}