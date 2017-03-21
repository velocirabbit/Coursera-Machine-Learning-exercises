function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad
%   X is of size m x n, y is of size m x 1, theta is of size n x 1

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly 
    %J = 0;
    grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
    % Get the linear model values. h = theta_0 + theta_1 * x
    h = X * theta;  % h is of size m x 1
    
    % Calculate the cost of this iteration
    J = (sum((h - y).^2) + lambda * sum(theta(2:end).^2)) / (2*m);
    
    % Get the gradient terms
    grad = (sum((h - y) .* X)' + lambda * [0; theta(2:end)]) / m;
% =========================================================================

    grad = grad(:);

end
