# Library file for ex5
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import warnings     # For catching RuntimeWarnings in fmincg()

# Compute the cost and gradient for regularized linear regresion with multiple
# variables.
def linearRegCostFunction(X, y, theta, lamb):
    m = np.alen(y)  # number of training examples
    theta = np.reshape(theta, [theta.size, 1])

    # Get the linear model values
    # h = theta_0 + theta_1 * x
    h = np.matmul(X, theta)  # h is of size [m x 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Calculate the cost of this iteration
        J = (np.sum((h - y)**2) + lamb * np.sum(theta[1:]**2)) / (2 * m)

        # Get the gradient terms
        grad = (np.sum((h - y) * X, axis = 0) + lamb * np.append(0, theta[1:])) / m
    grad = np.reshape(grad, [theta.size, 1])
    return [J, grad]

# Trains linear regression given a dataset (X, y) and a regularization
# parameter lambda.
def trainLinearReg(X, y, lamb):
    # Initialize theta
    initial_theta = np.zeros([X.shape[1], 1])

    # Create lambda function for the cost function to be minimized
    costFn = lambda t: linearRegCostFunction(X, y, t, lamb)

    # Minimize using fmincg
    print("\nTraining via fmincg...")
    options = {'MaxIter': 200, 'Print': False}
    theta_fmincg, cost_fmincg, __ = fmincg(costFn, initial_theta, options)

    # Minimize using op.minimize()
    # If jac = True, costFn is assumed to return the cost as the first return
    # value, and the gradient as the second.
    print("Training via op.minimize()...")
    res = op.minimize(fun = costFn, x0 = initial_theta, jac = True,
                        method = 'TNC', options = {'maxiter': 200})
    theta_opmin = res.x
    cost_opmin = res.fun

    if len(cost_fmincg) == 0:
        cost_fmincg = [np.nan]

    print("  Cost via fmincg:        %.6f" % cost_fmincg[-1])
    print("  Cost via op.minimize(): %.6f" % cost_opmin)

    if cost_fmincg[-1] < cost_opmin:
        theta = theta_fmincg
    else:
        theta = theta_opmin

    return theta  # trained theta values

# Generates the training and cross validation set errors needed to plot a
# learning curve
def learningCurve(X, y, Xval, yval, lamb):
    m = X.shape[0]

    # Pre-initialize vectors to store error data
    error_train = np.zeros([m, 1])
    error_val = np.zeros([m, 1])

    for i in range(m):
        iX = X[1:i, :]
        iY = y[1:i, :]

        # Get training error
        theta = trainLinearReg(iX, iY, lamb)
        error_train[i], __ = linearRegCostFunction(iX, iY, theta, 0)

        # Get cross validation error. Use the theta trained over the training
        # set
        error_val[i], __ = linearRegCostFunction(Xval, yval, theta, 0)

    return [error_train, error_val]

# Generate the training and validation errors needed to plot a validation curve
# that we can use to select lambda
def validationCurve(X, y, Xval, yval):
    # Selected values of lambda
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    numL = len(lambda_vec)

    error_train = np.zeros([numL, 1])
    error_val = np.zeros([numL, 1])

    for i in range(numL):
        lamb = lambda_vec[i]

        # Get theta parameters using this lambda
        theta = trainLinearReg(X, y, lamb)

        # Get training error. Note: don't find error of regularization term
        # (use lambda = 0)
        error_train[i], __ = linearRegCostFunction(X, y, theta, 0)

        # Get cross validation errors. Note: don't find error of regularization
        # term (use lambda = 0)
        error_val[i], __ = linearRegCostFunction(Xval, yval, theta, 0)

    return [lambda_vec, error_train, error_val]

# Maps X (a 1D vector) into the p-th power
def polyFeatures(X, p):
    X_poly = np.zeros([X.size, p])
    for i in range(p):
        X_poly[:, i] = np.transpose(X**(i + 1))
    return X_poly

# Normalizes the features in X
def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis = 0)
    X_norm = X_norm / sigma
    return [X_norm, mu, sigma]

# Plots a learned polynomial regression fit over an existing figure
def plotFit(min_x, max_x, mu, sigma, theta, p):
    # We plot a range slightly bigger than the min and max values to get an
    # idea of how the fit will vary outside the range of the data points
    x = np.transpose(np.arange(min_x - 15, max_x + 25, 0.05))

    # Map the X values
    X_poly = (polyFeatures(x, p) - mu) / sigma

    # Add a column of 1s
    X_poly = np.concatenate([
        np.ones([x.shape[0], 1]), X_poly
    ], axis = 1)

    # Plot
    plt.plot(x, np.matmul(X_poly, theta), '--', linewidth = 2)

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
def fmincg(f, X, options = None):
    ogshape = X.shape
    X = X.flatten()

    if options is not None:
        if 'MaxIter' in options:
            length = options['MaxIter']
        else:
            length = 100

        if 'Print' in options:
            printing = options['Print']
        else:
            printing = True

    # A bunch of constants for line searches
    RHO = 0.01      # RHO and SIG are the constants in the Wolfe-Powell conditions
    SIG = 0.5
    INT = 0.1       # Don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0       # Extrapolate maximum 3 times the current bracket
    MAX = 20        # Max 20 function evaluations per line search
    RATIO = 100     # Maximum allowed slope ratio

    # Build lambda of the function to minimize
    feval = lambda: f(X)

    # Check to see if the optional function reduction value is given
    if hasattr(length, 'shape') and np.max(length.shape) == 2:
        red = length[1]
        length = length[0]
    else:
        red = 1
    S = 'Iteration '    # for printing progress log to output

    i = 0               # Zero the run length counter
    ls_failed = 0       # No previous line search has failed
    fX = np.array([])
    f1, df1 = feval()                       # Get function value and gradient
    i += length < 0                         # Count epochs?!
    s = -df1                                # Search direction is steepest
    d1 = np.matmul(-np.transpose(s), s)     # This is the slope
    z1 = red / (1 - d1)                     # Initial step is: red/(|s| + 1)

    # Main body of optimization loop
    # Ignore RuntimeWarnings due to numerical errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while i < abs(length):                  # While not finished
            ### Set up ###
            i += length > 0                     # Count iterations?!

            X0 = X; f0 = f1; df0 = df1          # Copy current values
            # NOTE: Using X += ... modifies X in place, while X = X + ... doesn't.
            X = X + z1 * np.transpose(s)           # Begin line search
            f2, df2 = feval()
            i += length < 0
            d2 = np.matmul(np.transpose(df2), s)    # -Slope
            f3 = f1; d3 = d1; z3 = -z1          # Initialize point 3 = point 1
            if length > 0:
                M = MAX
            else:
                M = min(MAX, -length - i)
            success = 0; limit = -1             # Initialize quantities
            
            ### Minimization ###
            while True:
                while (f2 > f1 + z1*RHO*d1 or d2 > -SIG * d1) and (M > 0):
                    limit = z1      # Tighten the bracket
                    if f2 > f1:
                        # Quadratic fit
                        z2 = z3 - (0.5*d3*z3*z3)/(d3 * z3 + f2 - f3)
                    else:
                        # Cubic fit
                        A = 6*(f2 - f3)/z3 + 3*(d2 + d3)
                        B = 3*(f3 - f2) - z3*(d3 + 2*d2)
                        z2 = (np.sqrt(B**2 - A*d2*z3*z3) - B)/A
                    
                    # If we had a numerical problem, then bisect
                    if np.isnan(z2) or np.isinf(z2):
                        z2 = z3/2
                    
                    # Don't accept too close to limits
                    z2 = max(min(z2, INT*z3), (1 - INT)*z3)
                    z1 += z2        # Update the step
                    X = X + z2 * np.transpose(s)
                    f2, df2 = feval()
                    M -= 1; i += length < 0     # Count epochs?!
                    d2 = np.matmul(np.transpose(df2), s)    # -Slope
                    z3 -= z2        # z3 is now relative to the location of z2
                # end while
                if f2 > f1 + z1*RHO*d1 or d2 > -SIG*d1 or M == 0:
                    break           # This is a failure
                elif d2 > SIG*d1:
                    success = 1     # Success
                    break
                
                # Make cubic extrapolation
                A = 6*(f2 - f3)/z3 + 3*(d2 + d3)
                B = 3*(f3 - f2) - z3*(d3 + 2*d2)
                z2 = -d2*z3*z3/(B + np.sqrt(B*B - A*d2*z3*z3))

                if not np.isreal(z2) or np.isnan(z2) or np.isinf(z2) or z2 < 0:
                    # Numerical problem or wrong sign?
                    if limit < -0.5:        # If we have no upper limit...
                        z2 = z1*(EXT - 1)   # ... then extrapolate the maximum amount
                    else:
                        z2 = (limit - z1) / 2   # ... otherwise bisect
                elif limit > -0.5 and z2 + z1 > limit:
                    # Extrapolation beyond max?
                    z2 = (limit - z1) / 2
                elif limit < -0.5 and z2 + z1 > z1*EXT:
                    # Extrapolation beyond limit
                    z2 = z1*(EXT - 1.0)         # Set to extrapolation limit
                elif z2 < -z3*INT:
                    z2 = -z3*INT
                elif limit > -0.5 and z2 < (limit - z1)*(1.0 - INT):
                    # Too close to limit?
                    z2 = (limit - z1)*(1.0 - INT)
                
                f3 = f2; d3 = d2; z3 = -z2      # Set point 3 = point 2
                z1 += z2; X = X + z2*np.transpose(s)   # Update current estimates
                f2, df2 = feval()
                M -= 1; i += length < 0     # Count epochs?!
                d2 = np.matmul(np.transpose(df2), s)
            # end while
            if success:     # If line search succeeded
                f1 = f2
                if fX.size == 0:
                    fX = np.array([f1])
                else:
                    fX = np.append(fX, f1)
                if printing:
                    print("%s %4i | Cost: %4.6e" % (S, i, f1))
                tdf1 = np.matmul(np.transpose(df1), df1)
                # Polak-Ribiere direction
                s = (np.matmul(np.transpose(df2), df2) - tdf1)/(tdf1)*s - df2
                tmp = df1; df1 = df2; df2 = tmp     # Swap derivatives
                d2 = np.matmul(np.transpose(df1), s)
                if d2 > 0:      # New slope must be negative
                    s = -df1    # Otherwise use steepest direction
                    d2 = np.matmul(-np.transpose(s), s)
                # np.finfo(np.float64).tiny gets the smallest positive usable
                # number of type np.float64
                z1 *= min(RATIO, d1/(d2 - np.finfo(np.float64).tiny))
                d1 = d2
                ls_failed = 0   # This line search didn't fail
            else:
                X = X0; f1 = f0; df1 = df0  # Restore point from before failed line search
                if ls_failed or i > abs(length):
                    # Line search failed twice in a row
                    break
                tmp = df1; df1 = df2; df2 = tmp     # Swap derivatives
                s = -df1        # Try steepest
                d1 = np.matmul(-np.transpose(s), s)
                z1 = 1/(1 - d1)
                ls_failed = 1       # This line search failed
    # Finish optimization. Return values
    X = np.reshape(X, ogshape)
    return [X, fX, i]
    