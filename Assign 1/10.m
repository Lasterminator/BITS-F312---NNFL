data = xlsread('data3.xlsx');
data(:, 1:4) = (data(:, 1:4) - mean(data(:, 1:4)))./std(data(:, 1:4));
X = randperm(size(data, 1));
train_data = data(X(1:0.6*size(data, 1)), :);
test_data = data(X(0.6*size(data, 1)+1:end), :);
c1 = [];                            %Arrays to store data points of each class
c2 = [];                            %to store probability of being in class 1, 2
p1 = 0;
p2 = 0;
for i = 1:1:size(train_data, 1)
    switch train_data(i, 5)
        case 1
            c1 = [c1; train_data(i, :)];
            p1 = p1+1;
        case 2
            c2 = [c2; train_data(i, :)];
            p2 = p2+1;
    end
end
p1 = p1/size(train_data, 1);        %Probability of the 2 classes
p2 = p2/size(train_data, 1);
c_mean1 = mean(c1(:, 1:4))';        %Mean of the 2 classes
c_mean2 = mean(c2(:, 1:4))';
cov_matrix1 = c1(:, 1:4)'*c1(:, 1:4);%Covariance matrix
det1 = det(cov_matrix1);
cov_matrix2 = c2(:, 1:4)'*c2(:, 1:4);
det2 = det(cov_matrix2);
Y = zeros(size(test_data, 1), 1);
for i = 1:1:size(test_data, 1)
    x = test_data(i, 1:4)';
    Y1 = ((2*pi)^(-2))*(det1^(-0.5))*exp(-0.5*(x-c_mean1)'*pinv(cov_matrix1)*(x-c_mean1));
    Y2 = ((2*pi)^(-2))*(det2^(-0.5))*exp(-0.5*(x-c_mean2)'*pinv(cov_matrix2)*(x-c_mean2));
    Y_mat = [Y1/Y2 p2/p1];              %Calculating Liklihood Ratio
    [~, Y(i)] = max(Y_mat);             %determining class
end
%Calculating confusion matrix
TN = 0;
TP = 0;
FN = 0;
FP = 0;
for i = 1:1:size(Y, 1)
    switch Y(i)
        case 1
            if Y(i) == test_data(i, 5)
                TN = TN + 1;
            else
                FN = FN + 1;
            end
        case 2
            if Y(i) == test_data(i, 5)
                TP = TP + 1;
            else
                FP = FP + 1;
            end
    end
end
c_matrix = [TN FP; TP FN]
Sensitivity = TP/(TP + FN)
Specificity = TN/(TN + FP)
Accuracy = (TP + TN)/(TP + TN + FP + FN)
