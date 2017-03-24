featureNormalize = function(X) {
  mu = colMeans(X)
  X_norm = t(t(X) - mu)
  
  sigma = as.double(lapply(1:ncol(X), function(i) sd(X_norm[,i])))
  X_norm = t(t(X_norm) / sigma)
  return(list(X = X_norm, mu = mu, sigma = sigma))
}