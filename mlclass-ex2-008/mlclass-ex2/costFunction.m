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
%
% Note: grad should have the same dimensions as theta
%

for i = 1 :m,
  row = X(i,:); % [xi0 xi1 xi2]
  prediction = sigmoid(row * theta); % htheta(xi)
  actual = y(i); % yi

  % Cost Function
  J += (-actual * log(prediction)) - ((1 - actual) * log(1 - prediction));

  % Gradient
  cost = prediction - actual;
  grad += cost * row';
end

J /= m; 
grad /= m;

% =============================================================

end
