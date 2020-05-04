data = xlsread('data3.xlsx');
data(:, 1:4) = (data(:, 1:4) - mean(data(:, 1:4)))./std(data(:, 1:4));
data(:, 5) = data(:, 5) - 1;
X = randperm(size(data, 1));
train_data = [ones(0.6*size(data, 1), 1) data(X(1:0.6*size(data, 1)), :)];
test_data = [ones(0.4*size(data, 1), 1) data(X(0.6*size(data, 1) + 1:end), :)];
w = zeros(1, size(train_data, 2) - 1);      %Weight initialization with zero column
alpha = 0.01;                               %Learning rate
iter_nos = 100;
Y = train_data(:, 6);
for i  = 1:1:iter_nos
    G = logsig(w*train_data(:,1:5)')';      %Hypothesis calculation
    for j = 1:1:size(train_data, 2) - 1
        w(j) = w(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*train_data(:, j));       %Weight iterations
    end
end
TP = 0;
TN = 0;
FP = 0;
FN = 0;
P = logsig(w*test_data(:, 1:5)')';
P = 1*(P<0.5);
Y_new = test_data(:, 6);
for i = 1:size(P, 1)
    switch (P(i))
        case 0
            if P(i) == Y_new(i)
                TN = TN+1;
            else
                FN = FN+1;
            end
        case 1
            if P(i) == Y_new(i)
                TP = TP + 1;
            else
                FP = FP + 1;
            end
    end
end
C_matrix = [TN FP;FN TP]       %Confusion matrix
Sensitivity = TP/(TP + FN)     %Sensitivity, Specificity and Accuracy caluclation
Specificity = TN/(TN + FP)
Accuracy = (TN + TP)/(TP + TN + FP + FN)
