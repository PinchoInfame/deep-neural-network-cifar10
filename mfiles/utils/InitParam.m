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