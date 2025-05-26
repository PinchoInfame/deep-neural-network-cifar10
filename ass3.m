addpath cifar-10-matlab/cifar-10-batches-mat/;
clc 
close all

%reading data
% [X_tra, Y_tra, y_tra] = LoadBatch('cifar-10-batches-mat/data_batch_1.mat');
% [X_val, Y_val, y_val] = LoadBatch('cifar-10-batches-mat/data_batch_2.mat');

% Load batches of images and labels
[X_tra1, Y_tra1, y_tra1] = LoadBatch('cifar-10-batches-mat/data_batch_1.mat');
[X_tra2, Y_tra2, y_tra2] = LoadBatch('cifar-10-batches-mat/data_batch_2.mat');
[X_tra3, Y_tra3, y_tra3] = LoadBatch('cifar-10-batches-mat/data_batch_3.mat');
[X_tra4, Y_tra4, y_tra4] = LoadBatch('cifar-10-batches-mat/data_batch_4.mat');
[X_tra5, Y_tra5, y_tra5] = LoadBatch('cifar-10-batches-mat/data_batch_5.mat');
X_tra = [X_tra1 X_tra2 X_tra3 X_tra4 X_tra5];
Y_tra = [Y_tra1 Y_tra2 Y_tra3 Y_tra4 Y_tra5];
y_tra = [y_tra1; y_tra2; y_tra3; y_tra4; y_tra5];
[X_tes, Y_tes, y_tes] = LoadBatch('cifar-10-batches-mat/test_batch.mat');
K = size(Y_tra, 1);
d = size(X_tra, 1);
N = size(X_tra, 2);
%Pre-process raw data (Normalize)
mean_x_tra = mean(X_tra, 2);
std_x_tra = std(X_tra, 0, 2);
X_tra = X_tra - repmat(mean_x_tra, [1, size(X_tra, 2)]);
X_tra = X_tra ./ repmat(std_x_tra, [1, size(X_tra, 2)]);
% X_val = X_val - repmat(mean_x_tra, [1, size(X_val, 2)]);
% X_val = X_val ./ repmat(std_x_tra, [1, size(X_val, 2)]);
X_tes = X_tes - repmat(mean_x_tra, [1, size(X_tes, 2)]);
X_tes = X_tes ./ repmat(std_x_tra, [1, size(X_tes, 2)]);
%Initialize parameters
m = [50 50]; %can be a vector
%m= [50 30 20 20 10 10 10 10];
k = size(m,2)+1;
[W, b] = InitParam(m, K, d);
%check gradients
% ndim = 20;
% lambda = 0;
% n_batch = 1;
% W_red=W;
% W_red{1} = W{1}(:, 1:ndim);
% X_red = X_tra(1:ndim, 1:n_batch);
% Y_red = Y_tra(:, 1:n_batch);
% [P, h] = EvaluateClassifier(X_red, W_red, b, k);
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_red, Y_red, W_red, b, lambda, 1e-5, k);
% [grad_W, grad_b] = ComputeGradients(X_red, Y_red, P, W_red, lambda, h, k);
% ngrad_W=ngrad_W.';
% ngrad_b=ngrad_b.';
% eps = 10e-6;
% for i=1:length(W_red)
%     err_W{i}=abs(grad_W{i}-ngrad_W{i}) ./ max(eps, abs(grad_W{i})+abs(ngrad_W{i}));
%     err_b{i}=abs(grad_b{i}-ngrad_b{i}) ./ max(eps, abs(grad_b{i})+abs(ngrad_b{i}));
%     err_max_W{i} = max(max(err_W{i}));
%     err_max_b{i} = (max(err_b{i}));
% end

%Mini-batch gradient descent algorithm without bn
% lambda = 0.005;
% GDparams.eta_min = 1e-5;
% GDparams.eta_max = 1e-1;
% GDparams.n_batch = 100;
% GDparams.ns = 5*50000/GDparams.n_batch;
% GDparams.n_epochs = 25;
% [Wstar, bstar, J_train, J_val, Loss_train, Loss_val, acc_train, acc_val] = MiniBatchGD(X_tra, Y_tra, y_tra, X_tes, Y_tes, y_tes, GDparams, W, b, lambda, k);
% max_acc_train = max(acc_train);
% max_acc_val = max(acc_val);

% epochs = 1:GDparams.n_epochs;
% figure(1)
% plot(J_train)
% hold on
% plot(J_val)
% legend('training data cost', 'validation data cost')
% xlabel('epoch')
% ylabel('cost')
% hold off

% figure(2)
% plot(Loss_train)
% hold on
% plot(Loss_val)
% legend('training data loss', 'validation data loss')
% xlabel('epoch')
% ylabel('loss')
% hold off

% figure(3)
% plot(acc_train)
% hold on
% plot(acc_val)
% legend('training data accuracy', 'validation data accuracy')
% xlabel('epoch')
% ylabel('accuracy')
% hold off

%check gradients for the k-layer NN with batch normalization
% ndim = 20;
% lambda = 0;
% n_batch = 20;
% W_red=W;
% W_red{1} = W{1}(:, 1:ndim);
% X_red = X_tra(1:ndim, 1:n_batch);
% Y_red = Y_tra(:, 1:n_batch);
% [P, h, S, mu, v, S_hat] = EvaluateClassifier(X_red, W_red, b, k);
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_red, Y_red, W_red, b, lambda, 1e-6, k);
% [grad_W, grad_b] = ComputeGradients(X_red, Y_red, P, W_red, lambda, h, k, S, S_hat, mu, v);
% ngrad_W=ngrad_W.';
% ngrad_b=ngrad_b.';
% eps = 10e-5;
% for i=1:length(W_red)
%     err_W{i}=abs(grad_W{i}-ngrad_W{i}) ./ max(eps, abs(grad_W{i})+abs(ngrad_W{i}));
%     err_b{i}=abs(grad_b{i}-ngrad_b{i}) ./ max(eps, abs(grad_b{i})+abs(ngrad_b{i}));
%     err_max_W{i} = max(max(err_W{i}));
%     err_max_b{i} = (max(err_b{i}));
% end

%Mini-batch gradient descent algorithm with bn
lambda = 0.004749; 
GDparams.eta_min = 1e-5;
GDparams.eta_max = 1e-1;
GDparams.n_batch = 100;
GDparams.ns = 2*50000/GDparams.n_batch;
GDparams.n_cycles = 3;
%GDparams.n_epochs = floor((GDparams.n_cycles*GDparams.n_batch*GDparams.ns)/(GDparams.n_batch*100));
GDparams.n_epochs = 20;
GDparams.alpha = 0.9;
[Wstar, bstar, J_train, J_val, Loss_train, Loss_val, acc_train, acc_val] = MiniBatchGD(X_tra, Y_tra, y_tra, X_tes, Y_tes, y_tes, GDparams, W, b, lambda, k);
max_acc_train = max(acc_train);
max_acc_val = max(acc_val);

epochs = 1:GDparams.n_epochs;
figure(1)
plot(J_train)
hold on
plot(J_val)
legend('training data cost', 'validation data cost')
xlabel('epoch')
ylabel('cost')
hold off

figure(2)
plot(Loss_train)
hold on
plot(Loss_val)
legend('training data loss', 'validation data loss')
xlabel('epoch')
ylabel('loss')
hold off

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

function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = im2double(A.data);
    X = X.';
    y = A.labels + 1;
    Y = y == 1:10;
    Y = Y.';
end

function [W, b] = InitParam(m, K, d)
    sig = 1e-4;
    W{1} = (1/sqrt(d)) * randn(m(1), d);
    %W{1}=normrnd(0, 1/sqrt(d), m(1), d);
    %W{1}=normrnd(0, sig, m(1), d);
    b{1}=zeros(m(1), 1);
    n_hlayers = size(m,2);
    for i=2:n_hlayers
        W{i} = (1/sqrt(m(i-1))) * randn(m(i), m(i-1));
        %W{i}=normrnd(0, 1/sqrt(m(i-1)), m(i), m(i-1));
        %W{i}=normrnd(0, sig, m(i), m(i-1));
        b{i}=zeros(m(i), 1);
    end
    W{n_hlayers + 1} = (1/sqrt(m(n_hlayers))) * randn(K, m(n_hlayers));
    %W{n_hlayers+1}=normrnd(0, 1/sqrt(m(n_hlayers)), K, m(n_hlayers));
    %W{n_hlayers+1}=normrnd(0, sig, K, m(n_hlayers));
    b{n_hlayers+1}=zeros(K, 1);
end

function [P, h, S, mu, v, S_hat] = EvaluateClassifier(X, W, b, k, mu_av, v_av)
    nb = size(X, 2);
    ones_nb = ones(1, nb);
    eps = 1e-6;
    for i=1:k-1
        S{i}=W{i}*X+b{i}*ones_nb;
        %batch normalization (forward pass)
        if nargin<6
            mu{i} = mean(S{i}, 2);
            v{i} =  mean((S{i} - repmat(mu{i}, 1, size(S{i}, 2))).^2, 2);
            S_hat{i} = diag((v{i} + eps).^(-0.5))*(S{i} - repmat(mu{i}, 1, size(S{i}, 2)));
        else
            mu{i}=mu_av{i};
            v{i}=v_av{i};
            S_hat{i} = diag((v{i} + eps).^(-0.5))*(S{i} - repmat(mu{i}, 1, size(S{i}, 2)));
        end
        h{i}=max(0, S_hat{i});
        X=h{i};
    end
        S{k} = W{k}*X + b{k}*ones_nb;
        P = softmax(S{k});
end

function [J, loss] = ComputeCost(X, Y, W, b, lambda, k, mu_av, v_av)
    D = size(X, 2);
    if nargin<8
        [P, ~] = EvaluateClassifier(X, W, b, k);
    else 
        [P, ~] = EvaluateClassifier(X, W, b, k, mu_av, v_av);
    end
    py = sum((Y .* P), 1);
    loss = -sum(log(py))/D;
    reg_term = 0;
    for i=1:length(W)
        reg_term=reg_term+(lambda*sumsqr(W{i}));
    end
    J = loss + reg_term;
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, h, k, S, S_hat, mu, v)
    eps = 1e-6;
    G = -(Y-P);
    nb = size(X, 2);
    ones_nb = ones(nb, 1);
    grad_W{k}=(G*(h{k-1}.'))/nb + 2*lambda*W{k};
    grad_b{k}=(G*ones_nb)/nb;
    G = (W{k}.')*G;
    G = G.*(h{k-1}>0);
    for l = k-1: -1 : 1
        %Batch normalization (backward pass)
        sigma1 = ((v{l}+eps).^(-0.5)); 
        sigma2 = ((v{l}+eps).^(-1.5));
        G1 = G.*(sigma1*(ones_nb.'));
        G2 = G.*(sigma2*(ones_nb.'));
        D = S{l}-mu{l}*(ones_nb.');
        c = (G2.*D)*ones_nb;
        G = G1-((G1*ones_nb)*(ones_nb.'))/nb-(D.*(c*(ones_nb.')))/nb;
        if l==1
            grad_W{l} = (G*(X.'))/nb + 2*lambda*W{l};
        else
            grad_W{l} = (G*(h{l-1}.'))/nb + 2*lambda*W{l};
        end
        grad_b{l} = (G*ones_nb)/nb;
        if l>1
            G = (W{l}.')*G;
            G = G.*(h{l-1}>0);
        end
    end
end

function [Wstar, bstar, J_train, J_val, Loss_train, Loss_val, acc_train, acc_val, mu_av, v_av] = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, W, b, lambda, k)
mu_av = 0;
v_av = 0;
n_batch = GDparams.n_batch;
eta = GDparams.eta_min;
n_epochs = GDparams.n_epochs;
n = size(X, 2);
iteration = 0;
alpha = GDparams.alpha;
for i=1:n_epochs    
    for j=1:n/n_batch
        iteration = iteration+1;
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        [P, h, S, mu, v, S_hat] = EvaluateClassifier(Xbatch, W, b, k);
        %calculate moving average
        for l=1:k-1
            if j==1
                mu_av=mu;
                v_av=v;
            else
                mu_av{l}=alpha*mu_av{l}+(1-alpha)*mu{l};
                v_av{l}=alpha*v_av{l}+(1-alpha)*v{l};
            end
        end
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda, h, k, S, S_hat, mu, v);
        for ii=1:length(W)
            W{ii} = W{ii}-eta*grad_W{ii};
            b{ii} = b{ii}-eta*grad_b{ii};
        end
        eta = ComputeEta(GDparams, iteration);
        GDparams.eta = eta;
    end
    [J_train_i, Loss_train_i] = ComputeCost(X, Y, W, b, lambda, k, mu_av, v_av);
    [J_val_i, Loss_val_i] = ComputeCost(X_val, Y_val, W, b, lambda, k, mu_av, v_av);
    acc_train(i) = ComputeAccuracy(X, y, W, b, k, mu_av, v_av);
    acc_val(i) = ComputeAccuracy(X_val, y_val, W, b, k, mu_av, v_av);
    J_train(i) = J_train_i;
    Loss_train(i) = Loss_train_i;
    J_val(i) = J_val_i;
    Loss_val(i) = Loss_val_i;
end
Wstar = W;
bstar = b;
end

function acc = ComputeAccuracy(X, y, W, b, k, mu_av, v_av)
    D = size(X, 2);
    if nargin<7
        [P, ~] = EvaluateClassifier(X, W, b, k);
    else
        [P, ~] = EvaluateClassifier(X, W, b, k, mu_av, v_av);
    end
    [~, yg] = max(P); 
    yg = yg .';
    acc = sum(yg==y)/D; 
end

function eta = ComputeEta(GDparams, iteration)
    cycle = floor(1+iteration/(2*GDparams.ns));
    x = abs((iteration/GDparams.ns) - (2*cycle) + 1);
    eta = GDparams.eta_min + (GDparams.eta_max-GDparams.eta_min)*max(0, (1-x));
end

function [acc_train, acc_val] = SearchLambda(X, Y, y, X_val, Y_val, y_val, W, b, GDparams, k)
    n_lambdas = 10;
    l_min = -3;
    l_max = -2;
    filename = 'SearchFine1';
    file = fopen(filename, 'w');
    fprintf(file,'Results for Search Lambda with %0f different lambda values with l_max: %1f and l_min: %2f \n', n_lambdas, l_max, l_min);
    fprintf(file,'Stepsize: %1f, batchsize: %2f, epochs: %3f, cycles: %4f \n\n\n', GDparams.ns, GDparams.n_batch, GDparams.n_epochs, GDparams.n_cycles);
    
    for l=1:n_lambdas
        lambda_i = GenerateLambda(l_min, l_max);
        [Wstar, bstar, ~, ~, ~, ~, acc_train, acc_val] = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, W, b, lambda_i, k);
        acc_train_max = max(acc_train);
        acc_val_max = max(acc_val);
        fprintf(file,'lambda: %0f \n', lambda_i);
        fprintf(file,'%2f percent validation Accuracy \n', acc_val_max);
        fprintf(file,'%2f percent Train Accuracy \n\n', acc_train_max);
    end
    fclose(file);
end

function lambda = GenerateLambda(l_min, l_max)
    l = l_min + (l_max - l_min) * rand(1, 1);
    lambda = 10^l;
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h, k)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda, k);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda, k);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda, k);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda, k);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

function y = softmax(x)
    % Subtract max for numerical stability
    x = x - max(x, [], 1); 
    exp_x = exp(x);
    y = exp_x ./ sum(exp_x, 1);
end

function s = sumsqr(x)
    s = sum(x(:).^2);
end