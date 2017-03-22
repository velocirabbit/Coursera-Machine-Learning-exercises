featureNormalize = function(X) {
  mu = colMeans(X)
  sig = as.double(lapply(1:dim(X)[2], function(i) sd(X[,i])))
  X_norm = scale(X)
  return(list(X_norm[,], mu = mu, sigma = sig))
}