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