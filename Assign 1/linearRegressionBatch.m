data = xlsread('data.xlsx');                        
X = [ones(size(data, 1), 1) data(:, 1:2)];          
X(:, 2) = (X(:, 2) - mean(X(:, 2)))/std(X(:, 2));   
X(:, 3) = (X(:, 3) - mean(X(:, 3)))/std(X(:, 3));   %normalisation
Y = data(:, 3);                                     
Y = (Y - mean(Y))/std(Y);                           
w = [0 0 0];    %weights initialsation              
alpha = 0.001;  %learning rate                   
iter_nos = 100;                    
J = zeros(iter_nos+1, 1);          
W = zeros(iter_nos+1, 2);          
J(1) = evaluatecostfunction(X, Y, w);
W(1,1) = w(1); 
W(1,2) = w(2);
for i = 1:1:iter_nos
    H = (X*(w')) - Y;                           
    for j = 1:1:3
        w(j) = w(j) - alpha*((H')*X(:, j));     
    end
    J(i+1) = evaluatecostfunction(X, Y, w);
    W(i+1, 1) = w(2);
    W(i+1, 2) = w(3);
end
x = 0:1:iter_nos;
plot(x, J);
w1 = 0.4:-0.001:-0.2;
w2 = 1:-0.005:-0.2;
J1 = zeros(length(w1), length(w2));
for i = 1:1:length(w1)
    for j = 1:1:length(w2)
        J1(i, j) = evaluatecostfunction(X, Y, [0 w1(i) w2(j)]); 
    end
end
w

%graph plotting
figure;
plot3(W(:, 1), J, W(:, 2));
grid on;
figure;
contour(w2, w1, J1);
hold on;
plot(W(:, 2), W(:, 1));

%function for above used cost function
function J = evaluatecostfunction(X, Y, W)          
    J = 0;
    for i = 1:1:size(X, 1)
        J = J + (W*(X(i, :)') - Y(i))^2;           
    end
    J = J/2;
end
