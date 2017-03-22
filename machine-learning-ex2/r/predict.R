predict = function(theta, X) {
  m = dim(X)[1]
  p = rep(0, m)
  p[sigmoid(X %*% theta) >= 0.5] = 1
  return(p)
}