data = xlsread('data4.xlsx');
data(:, 1:7) = (data(:, 1:7) - mean(data(:, 1:7)))./std(data(:, 1:7));
X = randperm(size(data, 1));
train_data = [zeros(0.6*size(data, 1), 1) data(X(1:0.6*size(data, 1)),:)];
test_data = [zeros(0.4*size(data, 1), 1) data(X(0.6*size(data, 1)+1:end), :)];
iter_nos = 100;
alpha = 0.001;                 
t = train_data;
for i = 1:1:90
    if train_data(i, 9) == 1
        t(i, 9) = 0;
    else
        t(i, 9) = 1;
    end
end
w1 = rand(1, 8);                %initializing random weights
Y = t(:, 9);
for i = 1:1:iter_nos
    G = logsig(w1*t(:, 1:8)')'; %Hypothesis calculation
    for j = 1:1:8
        w1(j) = w1(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*t(:, j));     %Weight update
    end
end

t = train_data;
for i = 1:1:90
    if train_data(i, 9) == 2
        t(i, 9) = 0;
    else
        t(i, 9) = 1;
    end
end
w2 = rand(1, 8);
Y = t(:, 9);
for i = 1:1:iter_nos
    G = logsig(w2*t(:, 1:8)')';
    for j = 1:1:8
        w2(j) = w2(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*t(:, j));
    end
end

t = train_data;
for i = 1:1:90
    if train_data(i, 9) == 3
        t(i, 9) = 0;
    else
        t(i, 9) = 1;
    end
end
w3 = rand(1, 8);
Y = t(:, 9);                %Output values
for i = 1:1:iter_nos
    G = logsig(w3*t(:, 1:8)')';
    for j = 1:1:8
        w3(j) = w3(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*t(:, j));
    end
end
%Testing with test data
P1 = logsig(w1*test_data(:, 1:8)')';        %Predistion of 1 vs 2 vs 3 cyclically
P2 = logsig(w2*test_data(:, 1:8)')';
P3 = logsig(w3*test_data(:, 1:8)')';
P = [P1 P2 P3];
P0 = zeros(size(P1, 1), 1);
for i = 1:1:size(P1, 1)
    [~, P0(i)] = max(P(i, :));              %Prediction by maximum probability
end

c = zeros(3);
for i = 1:1:size(P0, 1)
    if test_data(i, 9) == 1
        c(1, P0(i)) = c(1, P0(i))+1;
    elseif test_data(i, 9) == 2
        c(2, P0(i)) = c(2, P0(i))+1;
    else
        c(3, P0(i)) = c(3, P0(i))+1;
    end
end
%Confusion matrix and Accuracy determination
IA = zeros(1,3);
OA = 0;
for i = 1:1:3
    IA(i) = c(i,i)/sum(c(i, :));
    OA = OA + c(i,i);
end
OA = OA/sum(c(:));

