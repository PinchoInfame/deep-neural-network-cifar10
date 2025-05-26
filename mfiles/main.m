% Add necessary paths for CIFAR-10 data and utility functions
% Ensure script runs from its own location
current_file_path = mfilename('fullpath');
[current_dir, ~, ~] = fileparts(current_file_path);
cd(current_dir);
addpath cifar-10-matlab/cifar-10-batches-mat/;
clc 
close all
addpath('utils')


% Load CIFAR-10 training batches and test batch using LoadBatch function
[X_tra1, Y_tra1, y_tra1] = LoadBatch('cifar-10-batches-mat/data_batch_1.mat');
[X_tra2, Y_tra2, y_tra2] = LoadBatch('cifar-10-batches-mat/data_batch_2.mat');
[X_tra3, Y_tra3, y_tra3] = LoadBatch('cifar-10-batches-mat/data_batch_3.mat');
[X_tra4, Y_tra4, y_tra4] = LoadBatch('cifar-10-batches-mat/data_batch_4.mat');
[X_tra5, Y_tra5, y_tra5] = LoadBatch('cifar-10-batches-mat/data_batch_5.mat');

% Concatenate all training batches into one dataset
X_tra = [X_tra1 X_tra2 X_tra3 X_tra4 X_tra5];
Y_tra = [Y_tra1 Y_tra2 Y_tra3 Y_tra4 Y_tra5];
y_tra = [y_tra1; y_tra2; y_tra3; y_tra4; y_tra5];
% Load the test batch
[X_tes, Y_tes, y_tes] = LoadBatch('cifar-10-batches-mat/test_batch.mat');

% Determine dataset dimensions:
% K = number of classes, d = input dimension, N = number of training samples
K = size(Y_tra, 1);
d = size(X_tra, 1);
N = size(X_tra, 2);

% Normalize the training data (zero mean and unit variance)
mean_x_tra = mean(X_tra, 2);
std_x_tra = std(X_tra, 0, 2);
X_tra = X_tra - repmat(mean_x_tra, [1, size(X_tra, 2)]);
X_tra = X_tra ./ repmat(std_x_tra, [1, size(X_tra, 2)]);

% Normalize the test data using training data mean and std
X_tes = X_tes - repmat(mean_x_tra, [1, size(X_tes, 2)]);
X_tes = X_tes ./ repmat(std_x_tra, [1, size(X_tes, 2)]);

% Define the architecture of the neural network:
% Here, two hidden layers with 50 neurons each
m = [50 50]; %can be a vector
%m= [50 30 20 20 10 10 10 10];
k = size(m,2)+1;   % number of layers including output layer

% Initialize weights and biases for the network
[W, b] = InitParam(m, K, d);

% Parameters for mini-batch gradient descent with batch normalization
lambda = 0.004749;   %obtained with a coarse-to-fine search
GDparams.eta_min = 1e-5;  % minimum learning rate
GDparams.eta_max = 1e-1;  % maximum learning rate
GDparams.n_batch = 100;   % mini-batch size
GDparams.ns = 2*50000/GDparams.n_batch;   % number of steps per cycle
GDparams.n_cycles = 3;    % number of learning rate cycles
GDparams.n_epochs = 20;   % total number of epochs to train
GDparams.alpha = 0.9;     % momentum parameter

% Train the neural network using mini-batch gradient descent
[Wstar, bstar, J_train, J_val, Loss_train, Loss_val, acc_train, acc_val] = MiniBatchGD(X_tra, Y_tra, y_tra, X_tes, Y_tes, y_tes, GDparams, W, b, lambda, k);

% Get maximum training and validation accuracy achieved
max_acc_train = max(acc_train);
max_acc_val = max(acc_val);

% Plot training and validation cost over epochs
epochs = 1:GDparams.n_epochs;
figure(1)
plot(J_train)
hold on
plot(J_val)
legend('training data cost', 'validation data cost')
xlabel('epoch')
ylabel('cost')
hold off

% Plot training and validation loss over epochs
figure(2)
plot(Loss_train)
hold on
plot(Loss_val)
legend('training data loss', 'validation data loss')
xlabel('epoch')
ylabel('loss')
hold off

% Plot training and validation accuracy over epochs
figure(3)
plot(acc_train)
hold on
plot(acc_val)
legend('training data accuracy', 'validation data accuracy')
xlabel('epoch')
ylabel('accuracy')
hold off

%Coarse to fine search for lambda
%GDparams.n_cycles = 2;
%GDparams.n_epochs = floor((GDparams.n_cycles*GDparams.n_batch*GDparams.ns)/(GDparams.n_batch*100));
%[acc_train, acc_val] = SearchLambda(X_tra, Y_tra, y_tra, X_tes, Y_tes, y_tes, W, b, GDparams, k);