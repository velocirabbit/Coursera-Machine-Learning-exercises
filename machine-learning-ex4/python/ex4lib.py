# Library file for ex4
import matplotlib.pyplot as plt
import numpy as np
import warnings     # For catching RuntimeWarnings in fmincg()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Returns the gradient of the sigmoid function evaluated at z
def sigmoidGradient(z):
    sig_func = sigmoid(z)
    return sig_func * (1 - sig_func)

def displayData(X, example_width = 0):
    if (example_width == 0):
        if (np.ndim(X) == 1):
            X = np.matrix(X)
        example_width = int(round(np.sqrt(X.shape[1])))

    cmap = 'gray'

    # Compute rows, cols
    m, n = X.shape
    example_height = int((n / example_width))

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between image padding
    pad = 1

    # Setup blank display
    # - in front for black borders in resulting image
    display_array = -np.ones([pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)])
    
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            
            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(abs(X[curr_ex,:]))
            hpad = pad + (j - 1) * (example_height + pad)
            wpad = pad + (i - 1) * (example_width + pad)
            eheight = np.array([hpad + h for h in range(example_height)])
            ewidth = np.array([wpad + w for w in range(example_width)])

            imgex = np.reshape(X[curr_ex,:], [example_width, example_height]) / max_val
            
            # Need to rotate image due to how MATLAB indexes things
            imgex = np.flipud(np.rot90(imgex))
            display_array[np.ix_(eheight, ewidth)] = imgex
            curr_ex += 1
        if curr_ex > m:
            break
    
    # Display image
    plt.imshow(display_array, vmin = -1, vmax = 1, cmap = cmap)

# Implements the neural network cost function for a two layer neural network 
# which performs classification
# nn_params should be of type ndarray
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                    num_labels, X, y, lamb, printing = False):
    # Unroll parameters
    paramunroll = hidden_layer_size * (input_layer_size + 1)
    theta1 = np.reshape(nn_params[:paramunroll], [hidden_layer_size, input_layer_size + 1])
    theta2 = np.reshape(nn_params[paramunroll:], [num_labels, hidden_layer_size + 1])

    m, n = X.shape
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    ### Initializations ###
    # Add a column of bias units to each example in the input training data
    X = np.concatenate([np.ones([m, 1]), X], axis = 1)

    # Initialize activation value matrix for the hidden layer
    a2 = np.zeros([m, hidden_layer_size + 1])
    k, __ = theta2.shape  # k is the number of classes
    # For each row of y, compares the sequence 1:k to y
    yBinK = np.arange(1, k + 1) == y

    if printing:
        print("  |", end = '')
    for i in range(m):
        ### Feedforward of the input data into the neural net ###
        # For training example i...
        # ... calculate the activation values of the hidden layer
        z2i = np.matmul(X[i,:], np.transpose(theta1))
        # Add a bias unit of value = 1 to the front
        a2i = np.append(1, sigmoid(z2i))
        a2[i,:] = a2i

        # ... calculate the output values
        z3i = np.matmul(a2i, np.transpose(theta2))
        aOuti = sigmoid(z3i)

        # ... get the cost of this example's output
        yk = yBinK[i,:]
        dOi = aOuti - yk

        ### Back propagation ###
        # ... propagate the error back to find the next error cost in using the
        # current theta1 values to calculate the hidden layer's activation vals
        d2i_b = np.matmul(dOi, theta2)  # Propagate output layer error back
        d2i = d2i_b[1:] * sigmoidGradient(z2i)  # gradient change

        # Accumulate the theta gradients
        # theta1_grad should be size [hidden_layer_size x n+1]
        theta1_grad = theta1_grad + np.matmul(np.transpose(np.array([d2i])),
                                              np.array([X[i,:]]))
        # theta2_grad should be size [num_classes x hidden_layer_size+1]
        theta2_grad = theta2_grad + np.matmul(np.transpose(np.array([dOi])),
                                              np.array([a2i]))
        
        if printing and (i % (m / 80) == 0):
            print("=", end = '')

    # Add bias unit to each layer
    bias1 = np.zeros([hidden_layer_size, 1])
    biasK = np.zeros([k, 1])
    bTheta1 = np.concatenate([bias1, theta1[:, 1:]], axis = 1)
    bTheta2 = np.concatenate([biasK, theta2[:, 1:]], axis = 1)

    # Get parameter gradients
    theta1_grad = (theta1_grad + lamb * bTheta1) / m
    theta2_grad = (theta2_grad + lamb * bTheta2) / m

    ### Get cost after a single iteration ###
    # Calculate the output values
    aOut = sigmoid(np.matmul(a2, np.transpose(theta2)))

    # Unregularized cost
    J = -np.sum(yBinK * np.log(aOut) + (1 - yBinK) * np.log(1 - aOut)) / m
    # Regularization term. First combine thetas
    allTheta = np.concatenate([theta1[:,1:], np.transpose(theta2[:,1:])], axis = 1)
    reg = lamb * np.sum(allTheta**2) / (2 * m)
    # Regularize the cost
    J += reg

    grad = np.append(theta1_grad.flatten(), theta2_grad.flatten())
    if printing:
        print("| Cost: %g" % J)
    return [J, grad]

# Randomly initialize the weights of a layer with L_in incoming connections and
# L_out outgoing connections
def randInitialWeights(L_in, L_out):
    # We base epsilon_init on the number of nodes in L_in and L_out using the
    # formula: epsilon_init = sqrt(6) / sqrt(L_in + L_out)
    epsilon_init = np.sqrt(6) / np.sqrt(L_in + L_out)
    rands = np.random.uniform(size = [L_out, L_in + 1])
    W = rands * 2 * epsilon_init - epsilon_init
    return W

# Initialize the weights of a layer with fan_in incoming connections and
# fan_out ongoing connections using a fixed strategy; this will help later in
# debugging
def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros([fan_out, fan_in + 1])
    W = np.reshape(np.sin(np.arange(1, W.size + 1)), W.shape) / 10
    return W

# Computes the gradient using "finite differences" and gives a numerical
# estimate of the gradient
def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e

        # Calculate perturbations and reshape thetas
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        #print(np.sum(negp), loss1, "||", np.sum(posp), loss2)
        # Compute the numerical gradient and reset perturbation vector
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad

# Creates a small neural network to check the backpropagation gradients
def checkNNGradients(lamb = 0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = np.transpose(np.array([1 + (np.arange(m) % num_labels)]))

    nn_params = np.append(theta1.flatten(), theta2.flatten())
    
    # Use lambdas to get shorthand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lamb)
    J, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations. The two columns you get
    # should be very similar
    print(np.concatenate([
        np.transpose(np.matrix(numgrad.flatten())),
        np.transpose(np.matrix(grad.flatten()))
    ], axis = 1))
    print("The above two columns should be very similar.")
    print("  (Left: numerical gradient || Right: analytical gradient)")

    # Evaluate the norm of the difference betwen two solutions.
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print("\nIf your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).")
    print("Relative difference: %g" % diff)

# Predict the label of an input given a trained neural network
def predict(theta1, theta2, X):
    m, n = X.shape
    num_labels = theta2.shape[0]

    h1 = sigmoid(
        np.matmul(np.concatenate([np.ones([m, 1]), X], axis = 1), np.transpose(theta1))
    )
    h2 = sigmoid(
        np.matmul(np.concatenate([np.ones([m, 1]), h1], axis = 1), np.transpose(theta2))
    )
    p = np.argmax(h2, axis = 1)
    return p + 1  # +1 because Python is 0-indexed
    
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
        else
            length = 100

        if 'Print' in options:
            printing = options['Print']
        else
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
    z1 = red / (1 - d1)                     # Initial ste is: red/(|s| + 1)

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
                        B = 3*(f3 - f2) * z3*(d3 + 2*d2)
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
                    # Num prob or wrong sign?
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
                X = X0; f1 = f0; df1 = df0  # Restore point form before failed line search
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
    