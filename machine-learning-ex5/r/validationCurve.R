validationCurve = function(X, y, Xval, yval) {
  lambda_vec = c(0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)
  numL = length(lambda_vec)
  
  error_train = rep(0, numL)
  error_val = rep(0, numL)
  
  for (i in 1:numL) {
    lambda = lambda_vec[i]
    
    # Get theta parameters using this lambda
    theta = trainLinearReg(X, y, lambda)
    
    # Get training error.
    # Note: don't find error of regularization term (use lambda = 0)
    errs = linearRegCostFunction(X, y, theta, 0)
    error_train[i] = errs$cost
    
    # Get cross validation errors
    # Note: don't find error of regularization term (use lambda = 0)
    errs = linearRegCostFunction(Xval, yval, theta, 0)
    error_val[i] = errs$cost
  }
  
  return(list(lambdas = lambda_vec, train = error_train, val = error_val))
}