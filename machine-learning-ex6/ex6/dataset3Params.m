function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

% parameters should try to search
param = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

x1 = [1 2 1]; x2 = [0 4 -1];

prediction_error = [];

for i = 1:length(param)
  for j = 1:length(param)
    C     = param(i);
    sigma = param(j);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions       = svmPredict(model, Xval);
    prediction_error  = [prediction_error ; mean(double(predictions ~= yval)) i j];
  end
end

disp("size of prediction_error is :");
disp(size(prediction_error));

[p_err index] = max(prediction_error(:, 1));

disp("max prediction_error is: ");
disp(p_err);
disp("index is :");
disp(index);
[p_err i j] = prediction_error(index, :);

C     = i;
sigma = j;
% =========================================================================

end
