% Adaline Linear Neuron function----------------------------------------------------
function [weights, W0, loss_history] = AdalineNeuron(fold, epochs, a, X_train, y_train)
    % Initializing weights and bias
    weights = randn(1, size(X_train, 2));
    W0 = 1; % bias of neuron
    for epoch = 1:epochs
        error = zeros(size(X_train, 1), 1); % Initialization of errors for this epoch
        for i = 1:size(X_train, 1)
            % Linear activation
            y(i) = X_train(i, :) * weights' + W0; % Linear activation function
            for j = 1:length(weights)
                weights(j) = weights(j) - a * (y(i) - y_train(i)) * X_train(i, j); % Online weight adjustment with the gradient descent method: Least Squares
            end
            error(i) = 0.5 * (y(i) - y_train(i))^2; % Squared Error for sample i
            W0 = W0 - a * (y(i) - y_train(i)); % Bias update
        end
        mse = mean(error); % Mean Squared Error for the current epoch
        loss_history(fold, epoch) = mse; % Storing the MSE for each epoch
    end
end