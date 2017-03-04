function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%;
% Note: grad should have th`e same dimensions as theta
%

%  J : computeCost
% neg = find(y == 0);

h = sigmoid(X*theta);

J = (-1 / m) * (sum(y' * log(h) + (1 - y)' * log(1 - h)));

% miss understand of grad
% grad is the lose of theta, that every iter.
% some param for gradient
alpha     = 0.01;% no need alpha
num_iters = 1500;% no need iter number

%{
for i = 1:num_iters
  theta = theta - (alpha / m) * X' * (sigmoid(X * theta) - y);
end
grad = theta;
%}
grad = (1 / m) * X' * (sigmoid(X * theta) - y);

% =============================================================

end
