data = xlsread('data.xlsx');
X = [ones(size(data, 1), 1) data(:, 1: 2)];
X(:, 2) = (X(:, 2) - mean(X(:, 2)))/(std(X(:, 2)));
X(:, 3) = (X(:, 3) - mean(X(:, 3)))/(std(X(:, 3)));
Y = data(:, 3);
Y = (Y - mean(Y))/(std(Y));
w = [0 0 0];                                 
alpha = 0.001;                               
iter_nos = 100;
t_instances = 50;                            
J = zeros(iter_nos*t_instances + 1, 1);      
W = zeros(iter_nos*t_instances + 1, 3);      
J(1) = evaluatecostfunction(X, Y, w);
W(1,1) = w(1);
W(1,2) = w(2);
W(1,3) = w(3);
for i = 1:1:iter_nos
    X_new = randperm(size(X, 1), t_instances);
    for j = 1:1:t_instances
        H = (X(X_new(j), :)*w') - Y(X_new(j));      
        for q = 1:1:3
            w(q) = w(q) - alpha*(H*X(X_new(j), q)); 
            W((i-1)*t_instances+j+1, q) = w(q);
        end
        J((i-1)*t_instances + j + 1) = evaluatecostfunction(X, Y, w);
    end
end
x = 0:1:iter_nos*t_instances;
plot(x,J)
w1 = 0.4:-0.001:-0.2;
w2 = 0.8:-0.005:-0.2;
J1 = zeros(length(w1), length(w2));
for i = 1:1:length(w1)
    for j = 1:1:length(w2)
        J1(i, j) = evaluatecostfunction(X, Y, [0 w1(i) w2(j)]);
    end
end
w

figure;
plot3(W(:, 2), W(:, 1), J)
grid on;
figure;
contour(w2, w1, J1);
hold on;
plot(W(:, 2), W(:, 1));

function J = evaluatecostfunction(X, Y, W)              
    J = 0;
    for i = 1:size(X, 1)
        J = J + ((W*X(i,:)') - Y(i))^2;
    end
    J = J/2;
end
