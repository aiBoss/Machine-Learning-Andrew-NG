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
          
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


z2= sigmoid([ones(m,1) X]*Theta1');
z3= sigmoid([ones(m,1) z2]*Theta2');
Y=zeros(size(z3));
for i=1:m,
Y(i,y(i))=1;
end;

J= ((-1/m)*sum(sum(Y.*log(z3)+(1-Y).*log(1-z3))))+((lambda/(2*m))*(sum(sum([Theta1(:,2:end)].^2))+sum(sum([Theta2(:,2:end)].^2))));

err3= z3-Y;
err2=(err3*Theta2).*sigmoidGradient([ones(m,1), [ones(m,1), X]*Theta1']);
err2=err2(:,2:end);
Theta1_grad=(1/m)*(err2'*[ones(m,1) X]);
Theta2_grad=(1/m)*(err3'*[ones(m,1) z2]);
Theta1_grad(:,2:end)+=(lambda/m)*[Theta1(:,2:end)];
Theta2_grad(:,2:end)+=(lambda/m)*[Theta2(:,2:end)];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
