function y = softmax(x)
    % Subtract max for numerical stability
    x = x - max(x, [], 1); 
    exp_x = exp(x);
    y = exp_x ./ sum(exp_x, 1);
end