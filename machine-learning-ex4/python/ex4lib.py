# Library file for ex4
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

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
    pprint([numgrad.flatten(), grad.flatten()])
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
    return p