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