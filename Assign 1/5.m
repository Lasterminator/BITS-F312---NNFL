data = xlsread('data.xlsx');
X = [ones(size(data, 1), 1) data(:, 1:2)];
X(:, 2) = (X(:,2) - mean(X(:, 2)))/std(X(:, 2));
X(:, 3) = (X(:, 3) - mean(X(:, 3)))/std(X(:, 3));
Y = data(:, 3);
Y = (Y - mean(Y))/std(Y);
w = [0 0 0];                               
alpha = 0.001;                             
lambda = 0.4;                              
iter_nos = 100;
J = zeros(iter_nos + 1, 1);                
W = zeros(iter_nos + 1, 2);               
J(1) = evaluatecostfunction(X, Y, w, lambda);
W(1, 1) = w(1);
W(1, 2) = w(2);
for i = 1:1:iter_nos
    H = X*(w)' - Y;                        
    for j = 1:1:3
        w(j) = w(j) - alpha*(((H')*X(:, j)) - lambda*w(j));
    end
    J(i+1) = evaluatecostfunction(X, Y, w, lambda);
    W(i+1, 1) = w(2);
    W(i+1, 2) = w(3);
end
x = 0:1:iter_nos;
plot(x, J);
w1 = 0.4:-0.001:-0.2;
w2 = 0.8:-0.005:-0.2;
J1 = zeros(length(w1), length(w2));
for i = 1:1:length(w1)
    for j = 1:1:length(w2)
        J1(i, j) = evaluatecostfunction(X, Y, [0 w1(i) w2(j)], lambda);
    end
end
w

figure;
plot3(W(:, 1), J, W(:, 2));
figure;
contour(w2, w1, J1);
hold on;
plot(W(:, 2), W(:, 1));

function J = evaluatecostfunction(X, Y, w, lambda)  
    J = 0;
    for i = 1:1:size(X, 1)
        J = J + (w*(X(i, :))' - Y(i))^2;
    end
    J = J + lambda*sum(w.^2);
    J = J/2;
end
