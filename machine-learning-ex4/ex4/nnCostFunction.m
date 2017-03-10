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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
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

% Theta1 is 25 by 401
% Theta2 is 10 by 26

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];

% y_vec is m by 10
y_vec = [];
for i = 1:m
    y_vec = [y_vec ; 1:num_labels == y(i)];
end

% a_1 is 5000 by 401
a_1 = X;

% z_2 is 25 by 5000
z_2 = Theta1 * a_1';

% a_2 is 26 by 5000
a_2 = [ones(1, size(z_2, 2)) ; sigmoid(z_2)];

% z_3 is 10 by 5000
z_3 = Theta2 * a_2;

% h and a_3 is 10 by 5000
h = a_3 = sigmoid(z_3);

h_vec = a_3(:);
% [N, I] = max(h, [], 2);

% p = I(:);

J = (1 / m) * sum(diag((-y_vec) * log(h) -  (1 - y_vec) * log(1 - h)));

punish_theta1 = power(Theta1(:,2:end), 2);
punish_theta2 = power(Theta2(:,2:end), 2);
% size(punish_theta1)
% size(punish_theta2)
punish = (lambda / (2 * m)) * (sum(sum(punish_theta1)) + sum(sum(punish_theta2)));

J = J + punish;
% -------------------------------------------------------------

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


delta_3 = a_3 - y_vec';
% size(delta_3) % 10 * 5000
% size(Theta2)  % 10 * 26
% size(z_2)     % 25 * 5000

% this .* operator will multi every element in Theta2 and delta_3   (one to one )
% beacuse Theta2'  has the same size of delta_3
delta_2 = Theta2' * delta_3 .* ...
    sigmoidGradient([ones(1, size(z_2,2)) ; z_2]);
% delta_2 is 25 * 5000
delta_2 = delta_2(2:end,:);

Theta1_grad = Theta1_grad + (delta_2 * a_1) ./ m;

% 10 * 26
Theta2_grad = Theta2_grad + delta_3 * a_2' ./ m;

% disp(size(Theta1_grad));
% disp(size(Theta2_grad));
% -------------------------------------------------------------

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
Theta1_grad_zero = Theta1_grad(:,1);
% 25 * 401
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta1_grad(:,1) = Theta1_grad_zero;


% 10 * 26
Theta2_grad_zero = Theta2_grad(:,1);
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;
Theta2_grad(:,1) = Theta2_grad_zero;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
