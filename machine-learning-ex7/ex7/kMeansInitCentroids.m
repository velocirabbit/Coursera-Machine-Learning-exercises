function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

    % You should return this values correctly
    centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%
    % Choose K random indices of training examples (out of m)
    i = randperm(size(X, 1), K); 
    
    % Set the initial centroid locations to these K examples
    centroids = X(i,:);
% =============================================================

end

