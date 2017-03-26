trainLinearReg = function(X, y, lambda) {
  # Initialize theta
  initial_theta = matrix(rep(0, ncol(X)), ncol = 1)
  
  # Minimize using fmincg:
  cat("\nTraining via fmincg...\n")
  options = c(MaxIter = 200, Print = F)
  costFn = function(t) linearRegCostFunction(X, y, t, lambda)
  fcg = fmincg(costFn, initial_theta, options)
  theta_fmincg = fcg$params
  cost_fmincg = fcg$cost
  
  # Minimize using optim:
  cat("\nTraining via optim...\n")
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
  theta_optim = res$par
  cost_optim = res$value
  
  if (length(cost_fmincg) == 0) {
    cost_fmincg = c(NaN)
  }
  
  cat("\n  Cost via fmincg:", cost_fmincg[length(cost_fmincg)])
  cat("\n  Cost via optim: ", cost_optim, "\n")
  
  if (cost_fmincg[length(cost_fmincg)] < cost_optim) {
    theta = theta_fmincg
  } else {
    theta = theta_optim
  }
  
  return(theta)
}