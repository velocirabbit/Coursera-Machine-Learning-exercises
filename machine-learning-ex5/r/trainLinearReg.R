trainLinearReg = function(X, y, lambda) {
  # Initialize theta
  initial_theta = rep(0, ncol(X))

  # Create a "short hand" anonymous function for the cost minimization
  getCostGrad = function(t, getNew) {
    # This saves the output from linearRegCostFunction into a global variable
    # getCG. getNew denotes whether or not we want to run linearRegCostFunction
    # again with p -- this will only happen when we want to get a new cost.
    if (getNew) {
      assign("getCG", linearRegCostFunction(X, y, t, lambda), envir = .GlobalEnv)
      return(getCG$cost)
    }
    else {
      return(getCG$grad)
    }
  }
  
  # More anonymous functions to get the cost and grad specifically
  getCost = function(p) getCostGrad(p, T)
  getGrad = function(p) getCostGrad(p, F)
  
  # Run the minimization optimization
  res = optim(initial_theta, fn = getCost, gr = getGrad,
              method = 'BFGS', control = list(maxit = 200))
  return(res$par)
}