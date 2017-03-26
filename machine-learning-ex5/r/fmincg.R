fmincg = function(f, X, options) {
# Minimize a continuous differentiable multivariate function. Starting point is
# given by "X" (size [D x 1]), and the function f must return a function value
# and a vector of partial derivatives, and take only one argument as input. The
# Polak-Ribiere flavor of conjugate gradients is used to compute search
# directions, and a line search using quadratic and cubic polynomial
# approximations and the Wolfe-Powell stopping criteria is used together with
# the slope ratio method for guessing initial step sizes. Additionally, a bunch
# of checks are made to make sure that exploration is taking place and that
# extrapolation will not be unboundedly large. The "len" gives the length of
# the run: if it is positive, it gives the maximum number of line searches. If
# it is negative, its aboslute gives the maximum allowed number of function
# evaluations. You can (optionally) give "len" a second component which will
# indicate the reduction in function value to be expected in the first line-
# search (defaults to 1.0). The function returns when either its length is up,
# or if no further progress can be made (i.e. we are at a minimum, or so close
# that due to numerical problems, we can't get any closer). If the function
# terminates within a few iterations, it could be an indication that the
# function value and derivatives are not consistent (i.e. there may be a bug in
# the implementation of the function f). This function returns the found 
# solution X, a vector of function values fX indicating the progress made,
# and the number of iterations (line searches or function evaluations,
# depending on the sign of "len") i used. options should be a dictionary of
# optimzation options. Gonna ignore the P1, ..., P5 that's present in the
# MATLAB implementation since it's never used.
  # Check for options
  if (!missing(options)) {
    if ('MaxIter' %in% names(options)) {
      len = options['MaxIter']
    } else {
      len = 100
    }
    
    if ('Print' %in% names(options)) {
      printing = options['Print']
    } else {
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
  
  # Check to see if the optional function reduction value is given
  if (length(len) == 2) {
    red = len[2]
    len = len[1]
  } else {
    red = 1
  }
  S = 'Iteration'  # for printing progress log to output
  
  i = 0           # Zero the run length counter
  ls_failed = 0   # No previous line search has failed
  fX = c()
  fret = feval()
  f1 = fret[[1]]; df1 = fret[[2]]   # Get funciton value and gradient
  i = i + (len < 0)           # Count epochs
  s = -df1                    # Search direction is steepest
  d1 = as.double(t(s) %*% s)  # This is the slope
  z1 = red / (1 - d1)         # Initial size is: red/(|s| + 1)
  
  # Main body of optimization loop
  while (i < abs(len)) {        # While not finished
    ### Set up ###
    i = i + (len > 0)        # Counter iterations?!
    
    X0 = X; f0 = f1; df0 = df1  # Copy current values
    X = sweep(X, MARGIN = 2, z1*s, '+')
    fret = feval()
    f2 = fret[[1]]; df2 = fret[[2]]
    i = i + (len < 0)
    d2 = as.double(t(df2) %*% s)           # -Slope
    f3 = f1; d3 = d1; z3 = -z1  # Initialize point 3 = point 1
    if (len > 0) {
      M = MAX
    } else {
      M = min(MAX, -len - i)
    }
    success = 0; limit = -1     # Initialize quantities
    
    ### Minimization ###
    while (T) {
      while ((f2 > f1 + z1*RHO*d1 || d2 > -SIG*d1) && (M > 0)) {
        limit = z1      # Tighten the bracket
        if (f2 > f1) {  # Quadratic fit
          z2 = z3 - (0.5*d3*z3*z3)/(d3*z3 + f2 - f3)
        } else {          # Cubic fit
          A = 6*(f2 - f3)/z3 + 3*(d2 + d3)
          B = 3*(f3 + f2) - z3*(d3 + 2*d2)
          z2 = (sqrt(B^2 - A*d2*z3*z3) - B)/A
        }
        
        # If we had a numerical problem, then bisect
        if (is.nan(z2) || is.infinite(z2)) {
          z2 = z3/2
        }
        
        # Don't accept too close to limits
        z2 = max(min(z2, INT*z3), (1 - INT)*z3)
        z1 = z1 + z2  # Update the step
        X = sweep(X, MARGIN = 2, z2*s, '+')
        fret = feval()
        f2 = fret[[1]]; df2 = fret[[2]]
        M = M - 1; i = i + (len < 0)  # Count epochs?!
        d2 = as.double(t(df2) %*% s)  # -Slope
        z3 = z3 - z2       # z3 is now relative to the location of z2
      }
      
      if (f2 > f1 + z1*RHO*d1 || d2 > -SIG*d1 || M == 0) {
        break        # This is a failure
      } else if (d2 > SIG*d1) {
        success = 1  # Success
        break
      }
      
      # Make cubic extrapolation
      A = 6*(f2 - f3)/z3 + 3*(d2 + d3)
      B = 3*(f3 - f2) - z3*(d3 + 2*d2)
      z2 = -d2*z3*z3/(B + sqrt(B*B - A*d2*z3*z3))
      
      if (!is.double(z2) || is.nan(z2) || is.infinite(z2) || z2 < 0) {
        # Numerical problem or wrong sign?
        if (limit < -0.5) {      # If we have no upper limit...
          z2 = z1*(EXT - 1)      # ... then extrapolate the maximum amount  
        } else {
          z2 = (limit - z1) / 2  # ... otherwise bisect
        }
      } else if (limit > -0.5 && z2 + z1 > limit) {
        # Extrapolation beyond max?
        z2 = (limit - z1) / 2
      } else if (limit < -0.5 && z2 + z1 > z1*EXT) {
        # Extrapolation beyond limit
        z2 = z1*(EXT - 1.0)      # Set to extrapolation limit
      } else if (z2 < -z3*INT) {
        z2 = -z3*INT
      } else if (limit > -0.5 && z2 < (limit - z1)*(1.0 - INT)) {
        # Too close to limit?
        z2 = (limit - z1)*(1.0 - INT)
      }
      
      f3 = f2; d3 = d2; z3 = -z2  # Set point 3 = point 2
      z1 = z1 + z2; X = sweep(X, MARGIN = 2, z2*s, '+')  # Update current estimates
      fret = feval()
      f2 = fret[[1]]; df2 = fret[[2]]
      M = M - 1; i = i + (len < 0)  # Count epochs?!
      d2 = as.double(t(df2) %*% s)
    }
    
    if (success) {  # If line search succeeded
      f1 = f2
      fX = append(fX, f1)
      if (printing) {
        cat(S, i, '| Cost:', f1, '\n')
      }
      # Polak-Ribiere direction
      s = as.double(((t(df2) %*% df2) - (t(df1) %*% df1))/(t(df1) %*% df1))*s - df2
      tmp = df1; df1 = df2; df2 = tmp  # Swap derivatives
      d2 = as.double(t(df1) %*% s)
      if (d2 > 0) {  # New slope must be negative
        s = -df1     # Otherwise use steepest direction
        d2 = as.double((-t(s)) %*% s)
      }
      # .Machine$double.xmin gets the smallest positive usable number
      z1 = z1 * min(RATIO, d1/(d2 - .Machine$double.xmin))
      d1 = d2
      ls_failed = 0  # This line search didn't fail
    } else {
      X = X0; f1 = f0; df1 = df0  # Restore point from before failed line search
      if (ls_failed || i > abs(len)) {  # Line search failed twice in a row
        break
      }
      tmp = df1; df1 = df2; df2 = tmp  # Swap derivatives
      s = -df1  # Try steepest
      d1 = as.double((-t(s)) %*% s)
      z1 = 1 / (1 - d1)
      ls_failed = 1  # This line search failed
    }
  }
  return(list(params = X, cost = fX, iters = i))
}