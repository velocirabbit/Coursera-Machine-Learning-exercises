import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def displayData(X, example_width = 0):
    if (example_width == 0):
        if (np.ndim(X) == 1):
            X = np.matrix(X)
            X = np.reshape(X, [X.shape[1], 1])
        example_width = int(round(np.sqrt(X.shape[1])))

    cmap = 'gray'

    # Compute rows, cols
    if (np.ndim(X) == 1):
        m = 1
        n = X.shape[0]
    else:
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
            max_val = max(abs(X[curr_ex,:]))
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

def lrcostfn(theta, X, y, lam):
    m, n = X.shape
    h = np.reshape(sigmoid(np.matmul(X, theta)), [m, 1])
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m + \
        lam * np.sum(theta[1:]**2) / (2 * m)

def lrgrad(theta, X, y, lam):
    m, n = X.shape
    grad = np.zeros(theta.shape)
    h = np.reshape(sigmoid(np.matmul(X, theta)), [m, 1])
    grad[0] = np.sum((h - y) * np.reshape(X[:,0], [m, 1])) / m
    gp = np.reshape(np.sum((h - y) * X[:,1:], axis = 0), [n - 1, 1])
    treg = np.reshape(lam * theta[1:], [n - 1, 1])
    grad[1:] = np.squeeze((gp + treg) / m)
    return grad

def lrCostFunction(theta, X, y, lam):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    h = np.reshape(sigmoid(np.matmul(X, theta)), [m, 1])
    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m + \
        lam * np.sum(theta[1:]**2) / (2 * m)
    
    grad[0] = np.sum((h - y) * np.reshape(X[:,0], [m, 1])) / m
    gp = np.sum((h - y) * X[:,1:], axis = 0)
    grad[1:] = (np.reshape(gp, [n - 1, 1]) + lam * theta[1:]) / m

    return [J, grad]

# Trains multiple logistic regression classifiers and returns all the
# classifiers in a matrix all_theta, where the i-th row of all_theta
# corresponds to the classifier for label i
def oneVsAll(X, y, num_labels, lamb):
    m, n = X.shape

    all_theta = np.zeros([num_labels, n + 1])

    # Add ones to the X data matrix
    X = np.concatenate([np.ones([m, 1]), X], axis = 1)

    # Set initial theta
    initial_theta = np.zeros([n + 1, 1])

    # Run op.minimize() to obtain the optimal theta
    # This function will return theta and the cost
    for c in range(num_labels):
        # Run the optimization once for each label using for the solutions y
        # whether or not each training example is equal to the label (y == c)
        res = op.minimize(fun = lrcostfn, x0 = initial_theta, args = (X, y == c, lamb),
                    method = 'BFGS', jac = lrgrad, options = {'maxiter': 400})
        all_theta[c,:] = res.x

    return all_theta

def predictOneVsAll(all_theta, X):
    m, num_labels = X.shape
    p = np.zeros([m, 1])

    # Add ones to the X data matrix
    X = np.concatenate([np.ones([m, 1]), X], axis = 1)
    p = np.argmax(sigmoid(np.matmul(X, np.transpose(all_theta))), axis = 1)

    return np.reshape(p, [m, 1])
    
# Predict the label of an input given a trained neural network
def predict(Theta1, Theta2, X):
    if (np.ndim(X) == 1):
        X = np.matrix(X)

    m, n = X.shape
    num_labels = Theta2.shape[0]

    p = np.zeros([m, 1])
    # Add bias unit to each trial X
    X = np.concatenate([np.ones([m, 1]), X], axis = 1)

    # Calculate hidden layer activations
    a2 = sigmoid(np.matmul(X, np.transpose(Theta1)))
    # Add bias unit to the hidden layer of each trial
    a2 = np.concatenate([np.ones([a2.shape[0], 1]), a2], axis = 1)
    # Calculate the output layer values and get the output node index with the
    # largest value - this is the prediction of the whole neural net
    # p is a vector of the indices of the largest values (what we want)
    p = np.argmax(sigmoid(np.matmul(a2, np.transpose(Theta2))), axis = 1)
    
    return p
