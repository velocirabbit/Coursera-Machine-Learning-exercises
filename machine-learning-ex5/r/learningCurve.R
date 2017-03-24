learningCurve = function(X, y, Xval, yval, lambda) {
  m = nrow(X)  # number of training examples
  
  error_train = rep(0, m)
  error_val = rep(0, m)

  for (i in 1:m) {
    iX = matrix(X[1:i,], ncol = ncol(X))
    iY = matrix(y[1:i,], ncol = 1)

    # Get training error
    theta = trainLinearReg(iX, iY, lambda)

    error_train[i] = linearRegCostFunction(iX, iY, theta, 0)$cost
    
    # Get cross validation error. Use the theta trained over the training set
    error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)$cost
  }
  return(list(train = error_train, val = error_val))
}