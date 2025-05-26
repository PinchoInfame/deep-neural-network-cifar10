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