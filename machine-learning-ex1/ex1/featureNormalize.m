function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
X_means = zeros(1, size(X, 2));
X_stds  = zeros(1, size(X, 2));

for i = 1:size(X,2)
    X_means(i) = mean(X(:, i));
end

for i = 1:size(X,2)
    X_stds(i) = std(X(:, i));
end

for i = 1:size(X,2)
    X_norm(:, i) = (X_norm(:, i) - X_means(i)) / X_stds(i);
end

for i = 1:size(X,2)
  mu(i)    = X_means(i);
  sigma(i) = X_stds(i);
 end

%disp(X_means);
%disp(X_stds);


%{
X1_mean = mean(X(:, 1));
X2_mean = mean(X(:, 2));
mu = [X1_mean, X2_mean];

X1_std = std(X(:, 1));
X2_std = std(X(:, 2));
sigma = [X1_std, X2_std];

X_norm(:, 1) = (X_norm(:, 1).- X1_mean)./X1_std;
X_norm(:, 2) = (X_norm(:, 2).- X2_mean)./X2_std;

}%

% ============================================================

end
