function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    sum0 = 0;
    sum1 = 0;
    for i = 1:length(y)
        sum0 = sum0 + (theta(1) + theta(2)* X(i, 2) - y(i)) * X(i, 1);
        sum1 = sum1 + (theta(1) + theta(2)* X(i, 2) - y(i)) * X(i, 2);
    end
    temp0 = theta(1) - alpha * (1 / m) * sum0;
    temp1 = theta(2) - alpha * (1 / m) * sum1;

    theta(1) = temp0;
    theta(2) = temp1;

    %{
    theta_temp = zeros(length(theta), 1);
    for index = 1:length(theta)
        theta_temp(index) = theta(index) - alpha * (1 / m) * ((theta' * X(index, :)' - y(index)) * X(index));
    end

    for index = 1:length(theta)
        theta(index) = theta_temp(index);
    end
    %}

    % ============================================================


    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % disp(J_history(iter));

end

end
