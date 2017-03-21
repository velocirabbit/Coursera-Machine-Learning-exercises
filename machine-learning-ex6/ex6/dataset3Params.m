function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
    % Vector of possible C and sigma vals
    factor = 3; np = 8;
    Cvals = 0.01 * (factor.^[0:np]);
    sigvals = 0.01 * (factor.^[0:np]);

    % You need to return the following variables correctly.
    C = 0;
    sigma = 0;
    
    minErr = nan;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
    for i = 1:np
        for j = 1:np
            % Train SVM model on a C and sigma value
            model = svmTrain(X, y, Cvals(i), ...
                            @(x1, x2) gaussianKernel(x1, x2, sigvals(j)));
            
            % Get predictions
            predictions = svmPredict(model, Xval);
            
            % Get prediction error
            err = mean(double(predictions ~= yval));
            
            % Update C and sigma if this err is better
            if (err < minErr || isnan(minErr))
                C = Cvals(i);
                sigma = sigvals(j);
                minErr = err;
            end
        end
    end
% =========================================================================
C
sigma
end
