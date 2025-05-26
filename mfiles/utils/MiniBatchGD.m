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