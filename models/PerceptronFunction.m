% Perceptron function----------------------------------------------------
function [weights, W0, loss_history] = PerceptronFunction(fold, epochs, a, X_train, y_train)
    % Initializing weights and bias
    weights = randn(1, size(X_train, 2));
    W0 = 1; % bias of neuron
    for epoch = 1:epochs
        error = zeros(size(X_train, 1), 1); % Initialization of errors for this epoch
        for i = 1:size(X_train, 1)
            y(i) = sign(X_train(i, :) * weights' + W0); % Hard-limiter activation
            if y(i) == 0
                y(i) = -1;
            end
            if y(i) ~= y_train(i)
                weights = weights + a * (y_train(i) - y(i)) * X_train(i, :); % online method: if missclassification -> weight update
                W0 = W0 + a * (y_train(i) - y(i)); % bias update
                error(i) = (y_train(i) - y(i))^2; % Squared Error for sample i
            end
        end
        mse = mean(error); % Mean Squared Error for the current epoch
        loss_history(fold, epoch) = mse; % Storing the MSE for each epoch
    end
end