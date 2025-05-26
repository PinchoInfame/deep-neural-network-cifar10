function lambda = GenerateLambda(l_min, l_max)
    l = l_min + (l_max - l_min) * rand(1, 1);
    lambda = 10^l;
end