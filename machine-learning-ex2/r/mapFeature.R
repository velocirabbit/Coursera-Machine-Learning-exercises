mapFeature = function(X1, X2) {
  degree = 6
  out = rep(1, length(X1))
  
  for (i in 1:degree) {
    for (j in 0:i) {
      o = (X1^(i - j)) * (X2^j)
      out = cbind(out, o)
    }
  }
  return(out)
}