data = xlsread('data4.xlsx');
data(:, 1:7) = (data(:, 1:7) - mean(data(:, 1:7)))./std(data(:, 1:7));
dataset1 = data(1:50,:);
dataset2 = data(51:100, :);
dataset3 = data(101:end, :);
X = randperm(50);
train_data = [dataset1(X(1:0.7*size(dataset1, 1)), :); dataset2(X(1:0.7*size(dataset2, 1)), :); dataset3(X(1:0.7*size(dataset3, 1)), :)];
test_data = [dataset1(X(0.7*size(dataset1, 1) + 1: end), :); dataset2(X(0.7*size(dataset2, 1) + 1: end), :); dataset3(X(0.7*size(dataset3, 1) + 1:end), :)];
t1 = []; t2 = []; t3 = [];              %Dividing classes based on probability
for i = 1:1:size(train_data, 1)
    switch train_data(i ,8)
        case 1
            t1 = [t1; train_data(i, :)];
        case 2
            t2 = [t2; train_data(i, :)];
        case 3
            t3 = [t3; train_data(i, :)];
    end
end
mean_t1 = mean(t1(:, 1:7))';            %Calculating the mean of 1,2,3
mean_t2 = mean(t2(:, 1:7))';
mean_t3 = mean(t3(:, 1:7))';
E1 = t1(:, 1:7)'*t1(:, 1:7);            %Calculating covariance matrix
E2 = t2(:, 1:7)'*t2(:, 1:7);
E3 = t3(:, 1:7)'*t3(:, 1:7);
det_E1 = det(E1);
det_E2 = det(E2);
det_E3 = det(E3);
Y = zeros(size(test_data, 1), 1);
for i = 1:1:size(test_data, 1)
    x = test_data(i, 1:7)';             %Calculating P(X/Y)
    P1 = ((2*pi)^(-2)*(det_E1^(-0.5))*exp(-0.5*(x-mean_t1)'*pinv(E1)*(x-mean_t1)));
    P2 = ((2*pi)^(-2)*(det_E2^(-0.5))*exp(-0.5*(x - mean_t2)'*pinv(E2)*(x - mean_t2)));
    P3 = ((2*pi)^(-2)*(det_E3^(-0.5))*exp(-0.5*(x - mean_t3)'*pinv(E3)*(x - mean_t3)));
    [~, Y(i)] = max([P1 P2 P3]);        %Calculating the ML Output
end
c = zeros(3);                           %Calculating the confusion matrix
for i = 1:1:size(Y, 1)
    if test_data(i, 8) == 1
        c(1, Y(i)) = c(1, Y(i)) + 1;
    elseif test_data(i, 8) == 2
        c(2, Y(i)) = c(2, Y(i)) + 1;
    elseif test_data(i, 8) == 3
        c(3, Y(i)) = c(3, Y(i)) + 1;
    end
end
IA = zeros(1, 3);
OA = 0;
for i = 1:1:3
    IA(i) = c(i,i)/sum(c(i, :));
    OA = OA + c(i, i);
end
OA = OA/sum(c(:))
IA
c
