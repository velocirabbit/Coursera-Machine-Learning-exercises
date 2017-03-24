function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

    % Reshape nn_params back into the parameters Theta1 and Theta2, the
    % weight matrices for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);

    % You need to return the following variables correctly 
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    %% Backpropagation
    % For each example in the input training data, run it feedforward
    % through the neural net, calculate the error, then backpropagate the
    % error in reverse through the neural net to modify the Theta1 and
    % Theta2 values.
    
    %%%%% Initializations %%%%%
    % Add a column of bias units to each example in the input training data
    X = [ones(m, 1), X];  % NOTE: bias units should be 1's!
    % Initialize activation value matrix for the hidden layer
    a2 = zeros(m, hidden_layer_size + 1);
    K = size(Theta2, 1);  % Number of classes
    yBinK = 1:K == y;  % For each row of y, compares the sequence 1:K to y-val
    
    for i = 1:m
        %%%%% Feedforward of the input data into the neural net %%%%%
        % For training example i....
        % ... calculate the activation values of the hidden layer
        z2i = X(i,:) * Theta1';
        a2i = [1, sigmoid(z2i)];
        a2(i,:) = a2i;  % Add a bias unit of value = 1 to the front
        
        % ... calculate the output values
        z3i = a2i * Theta2';
        aOuti = sigmoid(z3i);
        
        % ... get the cost of this example's output
        yk = yBinK(i,:);
        dOi = aOuti - yk;
        
        % ... propagate the error back to find the next error cost in using
        % the current Theta1 values to calculate the hidden layer's
        % activation values.
        d2i_b = dOi * Theta2;   % Propagate output layer error back
        d2i =  d2i_b(2:end) .* sigmoidGradient(z2i);  % gradient change
        
        % Accumulate the Theta gradients
        bias1 = zeros(hidden_layer_size, 1);
        Theta1_grad = Theta1_grad + d2i' * X(i,:);
        biasK = zeros(K, 1);
        Theta2_grad = Theta2_grad + dOi' * a2i ;
    end
    
    % Add regularization terms after backprop loop
    Theta1_grad = (Theta1_grad + lambda * [bias1, Theta1(:, 2:end)])/ m;
    Theta2_grad = (Theta2_grad + lambda * [biasK, Theta2(:, 2:end)]) / m;

    %% Feedforward of the input data into the neural net
    
    
%    % Calculate the activation values of Layer 2 (first hidden layer)
%    a2 = sigmoid(X * Theta1');
%    % Add a column of bias units to each example
%    a2 = [ones(m, 1), a2];  % NOTE: bias units should be 1's!
    
    % Calculate the output values
    aOut = sigmoid(a2 * Theta2');

    % Unregularized cost
    J = -sum(sum(yBinK .* log(aOut) + (1 - yBinK) .* log(1 - aOut))) / m;
    % Regularization term
    allTheta = [Theta1(:, 2:end), Theta2(:, 2:end)'];  % Combine Thetas
    reg = lambda * sum(sum(allTheta.^2)) / (2*m);          % for easier sums
    % Regularize the cost
    J = J + reg;
    
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
