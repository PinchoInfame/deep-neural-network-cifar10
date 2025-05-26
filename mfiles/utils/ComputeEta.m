function eta = ComputeEta(GDparams, iteration)
    cycle = floor(1+iteration/(2*GDparams.ns));
    x = abs((iteration/GDparams.ns) - (2*cycle) + 1);
    eta = GDparams.eta_min + (GDparams.eta_max-GDparams.eta_min)*max(0, (1-x));
end