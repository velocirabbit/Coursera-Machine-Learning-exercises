polyFeatures = function(X, p) {
  X_poly = matrix(0, nrow = length(X), ncol = p)
  
  for (i in 1:p) {
    X_poly[,i] = X^i
  }
  return(X_poly)
}