function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = X*theta;

p= sigmoid(h);

J = -(1/m)*[([log(p)]'*y)+([log(1-p)]'*(1-y))] + [(lambda/(2*m))*[sum([theta(2:length(theta))].^2)]];

grad(1) = (1/m)*[sum((p-y).*X(:,1))]';

grad(2:length(theta)) = (1/m)*[[sum((p-y).*X(:,2:length(theta)))]' + lambda*theta(2:length(theta))];


end
