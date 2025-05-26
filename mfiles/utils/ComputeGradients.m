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