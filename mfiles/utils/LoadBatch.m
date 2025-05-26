function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = im2double(A.data);
    X = X.';
    y = A.labels + 1;
    Y = y == 1:10;
    Y = Y.';
end