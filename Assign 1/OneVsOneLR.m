data = xlsread('data4.xlsx');
data(:, 1:7) = (data(:, 1:7) - mean(data(:, 1:7)))./std(data(:, 1:7));
X = randperm(size(data, 1));
train_data = [ones(0.6*size(data, 1), 1) data(X(1:0.6*size(data, 1)), :)];
test_data = [ones(0.4*size(data, 1) ,1) data(X(0.6*size(data, 1)+1:end), :)];
iter_nos = 500;
alpha = 0.0001;
w1 = zeros(1, 8);
w2 = zeros(1, 8);
w3 = zeros(1, 8);
t = [];
for i = 1:1:90
    if train_data(i ,9) == 1
        t = [t, train_data(i, 1:8) 0];
    elseif train_data(i, 9) == 1
        t = [t; train_data(i, 1:8) 1];
    end
end
Y = t(:, 9);
for i = 1:1:iter_nos
    G = logsig(w1*t(:, 1:8)')';
    for j = 1:1:8
        w1(j) = w1(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*t(:, j));
    end
end

t = [];
for i = 1:1:90
    if train_data(i,9) == 2
        t = [t;train_data(i, 1:8) 0];
    elseif train_data(i, 9) == 3
        t = [t;train_data(i, 1:8) 1];
    end
end
Y = t(:, 9);
for i = 1:1:iter_nos
    G = logsig(w2*t(:, 1:8)')';
    for j = 1:1:8
        w2(j) = w2(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*t(:, j));
    end
end

t = [];
for i = 1:1:90
    if train_data(i,9) == 3
        t = [t;train_data(i, 1:8) 0];
    elseif train_data(i, 9) == 1
        t = [t;train_data(i, 1:8) 1];
    end
end
Y = t(:, 9);
for i = 1:1:iter_nos
    G = logsig(w3*t(:, 1:8)')';
    for j = 1:1:8
        w3(j) = w3(j) - alpha*sum((Y.*(1-G) + (Y-1).*G).*t(:, j));
    end
end

P1 = logsig(w1*test_data(:, 1:8)')';
P2 = logsig(w2*test_data(:, 1:8)')';
P3 = logsig(w3*test_data(:, 1:8)')';
Y1 = zeros(size(P1, 1), 1);
for i = 1:1:size(P1, 1)
    if P1(i)>0.5
        Y1(i) = 2;
    else
        Y1(i) = 1;
    end
end
Y2 = zeros(size(P2, 1), 1);
for i = 1:1:size(P2, 1)
    if P2(i) > 0.5
        Y2(i) = 2;
    else
        Y2(i) = 3;
    end
end
Y3 = zeros(size(P3, 1), 1);
for i = 1:1:size(P3, 1)
    if P3(i) >0.5
        Y3(i) = 3;
    else
        Y3(i) = 1;
    end
end
Y = zeros(size(P1, 1), 1);
for i = 1:1:size(Y, 1)
    if (Y1(i) + Y2(i) + Y3(i)) ~= 6
        Y(i) = mode([P1(i) P2(i) P3(i)]);
    else
        Y_r = [P1(i) P2(i) P3(i)];
        [~, Y(i)] = max(Y_r);
        if Y(i) == 1
            t = 1;
        elseif Y(i) == 2
            t = 2;
        elseif Y(i) == 3
            t = 3;
        end
        Y(i) = t;
    end
end
c = zeros(3);
for i = 1:1:size(Y, 1)
    if test_data(i ,9) == 1
        c(1, Y(i)) = c(1, Y(i))+1;
    elseif test_data(i, 9) == 2
        c(2, Y(i)) = c(2, Y(i))+1;
    else
        c(3, Y(i)) = c(3, Y(i))+1;
    end
end
IA = zeros(1,3);
OA = 0;
for i = 1:1:3
    IA(i) = c(i,i)/sum(c(i, :));
    OA = OA + c(i,i);
end
OA = OA/sum(c(:))
c
w1
w2
w3
IA
