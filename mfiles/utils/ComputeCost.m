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