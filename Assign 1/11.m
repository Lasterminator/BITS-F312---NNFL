data = xlsread('data4.xlsx');
data(:, 1:7) = (data(:, 1:7) - mean(data(:, 1:7)))./std(data(:, 1:7));
data = [ones(length(data), 1) data];
X = randperm(length(data));
train_data = data(X(1:0.7*size(data, 1)), :);
test_data = data(X(0.7*size(data, 1) + 1:end), :);
t1 = [];t2 = [];t3 = [];                %Diving on the basis of output class
P1 = 0; P2 = 0; P3 = 0;
for i = 1:1:length(train_data)
    switch train_data(i, end)
        case 1
            t1 = [t1; train_data(i, :)];
            P1 = P1 + 1;
        case 2
            t2 = [t2; train_data(i, :)];
            P2 = P2 + 1;
        case 3
            t3 = [t3; train_data(i, :)];
            P3 = P3 + 1;
    end
end
P1 = P1/size(train_data, 1);              %Probability of classes
P2 = P2/size(train_data, 1);
P3 = P3/size(train_data, 1);
mean_t1 = mean(t1(:, 1:8))';                %Calculating means and covariance matrices
mean_t2 = mean(t2(:, 1:8))';
mean_t3 = mean(t3(:, 1:8))';
E1 = t1(:, 1:8)'*t1(:, 1:8);
E2 = t2(:, 1:8)'*t2(:, 1:8);
E3 = t3(:, 1:8)'*t3(:, 1:8);
det_E1 = det(E1); det_E2 = det(E2); det_E3 = det(E3);
Y = zeros(size(test_data, 1));
for i = 1:1:size(test_data, 1)
    x = test_data(i, 1:end-1)';
    p1 = ((2*pi)^(-2))*det_E1^(-0.5)*exp(-0.5*((x - mean_t1)'/E1)*(x - mean_t1));
    p2 = ((2*pi)^(-2))*det_E2^(-0.5)*exp(-0.5*((x - mean_t2)'/E2)*(x - mean_t2));
    p3 = ((2*pi)^(-2))*det_E3^(-0.5)*exp(-0.5*((x - mean_t3)'/E3)*(x - mean_t3));
    P = [p1*P1 P2*p2 P3*p3];                %calculating P(y/x)
    [~, Y(i)] = max([P]);                   %calculating MAP outputa
end
c = zeros(3);                               %Calculating the confusion matrix
for i = 1:1:size(Y, 1)
    c(test_data(i, end), Y(i)) = c(test_data(i, end), Y(i))+1;
end
IA = zeros(1,3);
OA = 0;
for i = 1:1:3
    IA(i) = c(i, i)/sum(c(i, :));
    OA = OA + c(i, i);
end
OA = OA/sum(c(:))
IA
c
