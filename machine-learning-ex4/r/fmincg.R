fmincg = function(f, X, options) {
# Minimize a continuous differentiable multivariate function. Starting point is
# given by "X" (size [D x 1]), and the function f must return a function value
# and a vector of partial derivatives, and take only one argument as input. The
# Polak-Ribiere flavor of conjugate gradients is used to compute search
# directions, and a line search using quadratic and cubic polynomial
# approximations and the Wolfe-Powell stopping criteria is used together with
# the slope ratio method for guessing initial step sizes. Additionally, a bunch
# of checks are made to make sure that exploration is taking place and that
# extrapolation will not be unboundedly large. The "length" gives the length of
# the run: if it is positive, it gives the maximum number of line searches. If
# it is negative, its aboslute gives the maximum allowed number of function
# evaluations. You can (optionally) give "length" a second component which will
# indicate the reduction in function value to be expected in the first line-
# search (defaults to 1.0). The function returns when either its length is up,
# or if no further progress can be made (i.e. we are at a minimum, or so close
# that due to numerical problems, we can't get any closer). If the function
# terminates within a few iterations, it could be an indication that the
# function value and derivatives are not consistent (i.e. there may be a bug in
# the implementation of the function f). This function returns the found 
# solution X, a vector of function values fX indicating the progress made,
# and the number of iterations (line searches or function evaluations,
# depending on the sign of "length") i used. options should be a dictionary of
# optimzation options. Gonna ignore the P1, ..., P5 that's present in the
# MATLAB implementation since it's never used.
  # Check for options
  if (!missing(options)) {
    if ('MaxIter' %in% names(options)) {
      len = options$MaxIter
    }
    else {
      len = 100
    }
    
    if ('Print' %in% names(options)) {
      printing = options$Print
    }
    else {
      printing = T
    }
  }
  
  # A bunch of constants for line searches
  RHO = 0.01    # RHO and SIG are the constants in the Wolfe-Powell conditions
  SIG = 0.5
  INT = 0.1     # Don't reevaluate within 0.1 of the limit of the current bracket
  EXT = 3.0     # Extrapolate maximum 3 times the current bracket
  MAX = 20      # Max 20 function evaluations per line search
  RATIO = 100   # Maximum allowed slope ratio
  
  # Build anonymous function of the function to minimize
  feval = function() f(X)
}