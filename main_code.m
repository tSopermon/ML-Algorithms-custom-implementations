clc;
close all;
clear all;
% Loading the file from the folder, excluding the 3 first text rows
fid = readtable('Iris.dat', 'NumHeaderLines', 3);
% SUBSET 1-----------------------------------------------------------------
A = fid(1:100, 1:7); % Extracting the data: first 100 rows and 7 columns
data1 = A{:, [1, 3]}; % Converting the columns [1, 3] to an array
% we want to normalize the data so that the mean is 0 and the standard deviation is 1
m1 = mean(data1); % Calculating the mean of all data
s1 = std(data1); % Calculating the standard deviation of all data
norm_data1 = (data1 - m1) ./ s1; % Normalizing data

% Repeat the same process for the other subsets
% SUBSET 2-----------------------------------------------------------------
B = fid(51:150, 1:7); 
data2 = B{:, [2, 3]};
m2 = mean(data2);
s2 = std(data2);
norm_data2 = (data2 - m2) ./ s2;

% SUBSET 3-----------------------------------------------------------------
C = load('subset3.mat');
m3 = mean(C.x);
s3 = std(C.x);
norm_data3 = (C.x - m3) ./ s3;

%--------------------------------------------------------------------------
% Visualizing the original and normalized data
figure('Name', 'Original and Normalized Data');
subplot(3, 2, 1);
scatter(data1(:, 1), data1(:, 2), 20, 'MarkerEdgeColor',[0 .5 .5], 'MarkerFaceColor',[0 .7 .7], 'LineWidth',1.5);
title('SUBSET 1')
subplot(3, 2, 2);
scatter(norm_data1(:, 1), norm_data1(:, 2), 20, 'MarkerEdgeColor',[0 .5 .5], 'MarkerFaceColor',[0 .7 .7], 'LineWidth',1.5);
title('Normalized SUBSET 1')

subplot(3, 2, 3);
scatter(data2(:, 1), data2(:, 2), 20, 'MarkerEdgeColor',[.5 .5 0], 'MarkerFaceColor',[.7 .7 0], 'LineWidth',1.5);
title('SUBSET 2')
subplot(3, 2, 4);
scatter(norm_data2(:, 1), norm_data2(:, 2), 20, 'MarkerEdgeColor',[.5 .5 0], 'MarkerFaceColor',[.7 .7 0], 'LineWidth',1.5);
title('Normalized SUBSET 2')

subplot(3, 2, 5);
scatter(C.x(:, 1), C.x(:, 2), 20, 'MarkerEdgeColor',[.5 0 .5], 'MarkerFaceColor',[.7 0 .7], 'LineWidth',1.5);
title('SUBSET 3')
subplot(3, 2, 6);
scatter(norm_data3(:, 1), norm_data3(:, 2), 20, 'MarkerEdgeColor',[.5 0 .5], 'MarkerFaceColor',[.7 0 .7], 'LineWidth',1.5);
title('Normalized SUBSET 3')
%--------------------------------------------------------------------------

% Selecting the subset to feed to our model---------------------------------------------------------
choise = menu('Choose the subset you want to use:', 'Subset 1', 'Subset 2', 'Subset 3');
switch choise
    case 1
        dataset = norm_data1;
    case 2
        dataset = norm_data2;
    case 3
        dataset = norm_data3;
end

% Initializing the outputs of the model
labels = [ones(50, 1); -ones(50, 1)]; 

% Splitting data into 2 subsets and suffling the data
X1 = dataset(1:50, :);
X2 = dataset(51:100, :);
y1 = labels(1:50, :);
y2 = labels(51:100, :);

% Generating random the indices for each class
idx1 = randperm(50);
idx2 = randperm(50);

% Shuffling the data
X1 = X1(idx1, :); 
X2 = X2(idx2, :);

%--------------------------------------------------------------------------
% Cross Validation with k-folds-------------------------------------------------
k = 10; % Number of folds

% To split each category into k subsets
subsets_X1 = cell(k, 1); % Initialization of 10 cell arrays to store the subsets of class 1
subsets_X2 = cell(k, 1); % Initialization of 10 cell arrays to store the subsets of class 2
combined = cell(k, 1); % Combination of subsets

for i = 1:k
% splitting the class 1 into k equal sets of 5 rows and 2 columns
    subsets_X1{i} = X1((i-1)*5 + 1:i*5, :); 
    subsets_X2{i} = X2((i-1)*5 + 1:i*5, :); 
% splitting the labels of each class into 10 equal sets of 5 rows and 1 column
    subsets_y1 = y1((i-1)*5 + 1:i*5, :); 
    subsets_y2 = y2((i-1)*5 + 1:i*5, :);
% combining the sets oo each class making 10 sets of 10 rows and 2 columns
    combined{i} = [subsets_X1{i}, subsets_y1; subsets_X2{i}, subsets_y2];
end
%--------------------------------------------------------------------------

% Choosing which model to use---------------------------------------------------------
choise = menu('Choose the model you want to use:', 'Adaline Linear Neuron', 'Logistic Regressor', 'Perceptron');
switch choise
    case 1
        model = 'Adaline Linear Neuron';
    case 2
        model = 'Logistic Regressor';
    case 3
        model = 'Perceptron';
end

% k-fold Cross Validation training-------------------------------------------------
% Selecting the learning rate of the model
choise = menu('Choose learning rate of model:', '0.01', '0.1', '0.5');
switch choise
    case 1
        a = 0.01;
    case 2
        a = 0.1;
    case 3
        a = 0.5;
end

% input the number of epochs
epochs = input('Enter the number of epochs: ');

% Initializing an empty table to store the MSE for each fold
mse_table = zeros(k, 1); % The table where we will store MSEs for each fold
min_mse = inf; % Initializing the minimum MSE to infinity
max_mse = -inf; % Initializing the maximum MSE to -infinity

max_accuarcy = 0; % Initializing the maximum accuracy to 0
min_accuracy = 0; % Initializing the minimum accuracy to 0

loss_history = zeros(1, epochs); % Initialization of loss history to store the MSE for each epoch
accu_table = zeros(k, 1); % Initialization of accuracy table to store the accuracy for each fold
bestHistory = zeros(1, epochs); % Initialization of best model loss history
worstHistory = zeros(1, epochs); % Initialization of worst model loss history

% Looping through each fold
for fold = 1:k
    % splitting training and test data
    % 70% will be used as training set
    % 30% will be used as testing set
    test_indices = mod((fold-1):(fold+1), 10) + 1; % mod function to get the indices of the test cells for each fold
    % SOS: by using mod we achieve a circular shift of the indices
    % Remaining cells for training
    train_indices = setdiff(1:10, test_indices);

    train_data = cell2mat(combined(train_indices)); % Training dataset
    test_data = cell2mat(combined(test_indices));
    % Extracting the features and labels
    X_train = train_data(:, 1:2); 
    y_train = train_data(:, 3);

    X_test = test_data(:, 1:2); 
    y_test = test_data(:, 3);

    y = zeros(size(train_data(:, 3))); % preallocating for speed
    y_pred = zeros(size(test_data(:, 3))); % preallocating for speed

    %--------------------------------------------------------------------------
    % Model Activation
    % if statements for activating function/model depending on the user's choice
    if strcmp(model, 'Adaline Linear Neuron')
        [weights, W0, loss_history] = AdalineNeuron(fold, epochs, a, X_train, y_train);
    elseif strcmp(model, 'Logistic Regressor')
        [weights, W0, loss_history] = LogisticRegressorFunction(fold, epochs, a, X_train, y_train);
    elseif strcmp(model, 'Perceptron')
        [weights, W0, loss_history] = PerceptronFunction(fold, epochs, a, X_train, y_train);
    end

    fold_mse = mean(loss_history(fold, :)); % mean of MSE across epochs for the current fold
    mse_table(fold) = fold_mse; % storing the mse for each fold, to extract later the max and min MSE

    % Making predictions with the test dataset, depending on the model
    for i = 1:size(X_test, 1)
        if strcmp(model, 'Adaline Linear Neuron')
            y_pred(i) = X_test(i,:) * weights' + W0; % Least Squares method for adaline linear neuron
        elseif strcmp(model, 'Logistic Regressor')
            y_pred(i) = tanh(X_test(i,:) * weights' + W0); % Hyperbolic tangent activation method for logistic regressor
        elseif strcmp(model, 'Perceptron')
            y_pred(i) = sign(X_test(i, :) * weights' + W0); % Hard-limiter activation method for perceptron
            if y_pred(i) == 0
                y_pred(i) = -1; % ensuring that the output is -1 if the output is 0
            end
        end
    end

    % Calculating the accuracy of the model
    accuracy = 100 * (1 - mean(abs(y_test - y_pred), "all")); 
    accu_table(fold) = accuracy; % storing the accuracy for each fold to extract the mean, min and max accuracy

    % store best model parameters
    if fold_mse < min_mse % if the current fold MSE is less than the minimum MSE
        min_mse = fold_mse;
        best_weights = weights;
        best_bias = W0;
        max_accuarcy = accuracy;
        best_loss_history = loss_history(fold, :); % storing loss history for visualization
    end

    % store worst model parameters
    if fold_mse > max_mse % if the current fold MSE is greater than the maximum MSE
        max_mse = fold_mse;
        worst_weights = weights;
        worst_bias = W0;
        min_accuracy = accuracy;
        worst_loss_history = loss_history(fold, :); % storing loss history for visualization
    end
    
    fprintf('Fold %d: Mean Squared Error = %.4f\n', fold, fold_mse);
    
end

% Displaying the results of the model
mean_accuracy = mean(accu_table, "all");
fprintf('\nMean Accuracy with K-fold cross validation = %.4f\n', mean_accuracy);
fprintf('\nBest Model Results:\nMSE = %.4f\nPredictions Accuracy = %.4f\n', min_mse, max_accuarcy);
fprintf('\nWorst Model Results:\nMSE = %.4f\nPredictions Accuracy = %.4f\n', max_mse, min_accuracy);

% Plotting the loss history of best and worst models separately in two figures with grid
figure("Name", "Loss History of Best Model");
plot(best_loss_history, 'g-', 'DisplayName', 'Best Model Loss History');
xlabel('Epochs');
ylabel('Mean Squared Error');
grid on
title('Loss History of Best Model');

figure("Name", "Loss History of Worst Model");
plot(worst_loss_history, 'r-', 'DisplayName', 'Worst Model Loss History');
xlabel('Epochs');
ylabel('Mean Squared Error');
grid on
title('Loss History of Worst Model');

% Plotting the decision boundaries of best and worst models
figure("Name", "Decision Boundaries of Best and Worst Models");
hold on;
title('Perceptron Model with best results');
xlabel('First Feature');
ylabel('Second Feature'); 
% Plot data points 
scatter(dataset(:, 1), dataset(:, 2), 'filled')
grid on
% Plot decision boundary
x_values = linspace(min(X_train(:, 1)), max(X_train(:, 1)), 100); % Generating 100 points between the minimum and maximum values of the first feature
ybest_values = -(best_weights(1) * x_values + best_bias) / best_weights(2); % Calculating the decision boundary of the best model according to the weights and bias we stored
yworst_values = -(worst_weights(1) * x_values + worst_bias) / worst_weights(2); % Same for the worst model
plot(x_values, ybest_values, 'g-', 'DisplayName', 'Decision Boundary of Best Model'); 
plot(x_values, yworst_values, 'r-', 'DisplayName', 'Decision Boundary of Worst Model'); 
axis([min(dataset(:, 1)) max(dataset(:, 1)) min(dataset(:, 2)) max(dataset(:, 2))]); % Set axis limits to match data 
legend("BackgroundAlpha", 0.5);
hold off;


% FUNCTIONS------------------------------------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------------------------------------------------------

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


% Logistic Regressor function----------------------------------------------------
function [weights, W0, loss_history] = LogisticRegressorFunction(fold, epochs, a, X_train, y_train)
    % Initializing weights and bias
    weights = randn(1, size(X_train, 2));
    W0 = 1; % bias of neuron
    for epoch = 1:epochs
        error = zeros(size(X_train, 1), 1); % Initialization of errors for this epoch
        for i = 1:size(X_train, 1)
            y(i) = tanh(X_train(i, :) * weights' + W0); % hyperbolic tangent activation function tanh
            for j = 1:length(weights)
                % Online weight adjustment with the gradient descent method using the derivative of tanh 
                weights(j) = weights(j) - a * (y(i) - y_train(i)) * (1 + y(i)) * (1 - y(i)) * X_train(i, j);
            end
            error(i) = 0.5 * (y(i) - y_train(i))^2; % Squared Error for sample i
            W0 = W0 - a * (y(i) - y_train(i)) * (1 + y(i)) * (1 - y(i)); % Bias update using the derivative of tanh
        end
        mse = mean(error); % Mean Squared Error for the current epoch
        loss_history(fold, epoch) = mse; % Storing the MSE for each epoch
    end
end


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