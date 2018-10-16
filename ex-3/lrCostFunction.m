function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


p= sigmoid(X*theta);

J = -(1/m)*[([log(p)]'*y)+([log(1-p)]'*(1-y))] + [(lambda/(2*m))*[sum([theta(2:end)].^2)]];

grad(1) = (1/m)*(X'(1,:)*(p-y));

grad(2:end) = (1/m)*[(X'(2:end,:)*(p-y))+ lambda*theta(2:end)];

grad = grad(:);

end
