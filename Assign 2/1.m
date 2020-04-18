clc;
clear;
S = load('data5.mat');
X = S.x;
for i = 1:72
    X(:,i) = (X(:,i)-mean(X(:,i)))./std(X(:,i));
end
size_data = size(X);%num of training examples = 2148
n0 = size_data(2) - 1; %num of ip features = 72
n1 = 36; % num of neurons in hidden layer 1
n2 = 12; % num of neurons in hidden layer 2
n3 = 2; % num of neurons in the output layer
w1 = rand(n1, n0).*0.1;
b1 = zeros(n1, 1);
w2 = rand(n2, n1).*0.1;
b2 = zeros(n2, 1);
w3 = rand(n3, n2).*0.1;
b3 = zeros(n3, 1);
dw1 = zeros(n1, n0);
db1 = zeros(n1, 1);
dw2 = zeros(n2, n1);
db2 =zeros(n2, 1);
dw3 = zeros(n3, n2);
db3 = zeros(n3, 1);
num_iter = 200;
J = zeros(num_iter, 1);
Y = X(:,73);
alpha = 0.001;
%1504 training examples(70%) and 644 testing examples
m = 1504;
xtrain = zeros(1504, 72);ytrain = zeros(2, 1504);
xtrain = X([1:752 ,1074:1825],1:72);
y1 = Y([1:752 ,1074:1825]);
y1 = y1';
ytrain(1, :) = y1;
for var = 1 : 1504
    if ytrain(1, var) == 1
    ytrain(2, var) = 0;
    else
    ytrain(2, var) = 1;
    end
end
xtest = X([753 : 1073 , 1826 : 2148], 1 : 72);
ytest = Y([753 : 1073 , 1826 : 2148]);

% TRAINING
for t = 1 : num_iter
    dw1=zeros(n1,n0);db1=zeros(n1,1);dw2=zeros(n2,n1);db2=zeros(n2,1);dw3=zeros(n3,n2);db3=zeros(n3,1);
    j = 0;
    for i = 1 : 1504
        z1=zeros(n1,1);a1=zeros(n1,1);z2=zeros(n2,1);a2=zeros(n2,1);z3=zeros(n3,1);a3=zeros(n3,1);
        for q = 1 : n1
            for r = 1 : n0
                z1(q)=z1(q)+w1(q,r)*xtrain(i,r);
            end
            z1(q)=z1(q)+b1(q);
        end
        a1=sigmoidFunction(z1);
        for p=1:n2
            for q=1:n1
                z2(p)=z2(p)+w2(p,q)*a1(q);
            end
            z2(p)=z2(p)+b2(p);
        end
        a2=sigmoidFunction(z2);
        for k=1:n3
            for p=1:n2
                z3(k)=z3(k)+w3(k,p)*a2(p);
            end
            z3(k) = z3(k) + b3(k);
        end
        a3 = sigmoidFunction(z3);
        for k = 1 : n3
            j = j + ((ytrain(k, i) - a3(k))^2);
        end
        for k = 1 : n3
            db3(k) = db3(k) - ((ytrain(k, i) - a3(k))*z3(k)*(1 -z3(k)));
        end
        for k = 1 : n3
            for p = 1 : n2
                dw3(k,p)=dw3(k,p)-((ytrain(k,i)-a3(k))*z3(k)*(1-z3(k))*a2(p));
            end
        end
        for p = 1 : n2
            for k = 1 : n3
                db2(p)=db2(p)-((ytrain(k,i)-a3(k))*z3(k)*(1-z3(k))*w3(k,p)*z2(p)*(1-z2(p)));
            end
        end
        for p = 1 : n2
            for q = 1 : n1
                for k = 1 : n3
                    dw2(p,q)=dw2(p,q)-((ytrain(k,i)-a3(k))*z3(k)*(1-z3(k))*w3(k,p)*z2(p)*(1-z2(p))*a1(q));
                end
            end
        end
        for q = 1 : n1
            for k = 1 : n3
                for p = 1 : n2
                    db1(q)=db1(q)-((ytrain(k,i)-a3(k))*z3(k)*(1-z3(k))*w3(k,p)*z2(p)*(1-z2(p))*w2(p, q)*z1(q)*(1-z1(q)));
                end
            end
        end
        for q = 1 : n1
            for r = 1 : n0
                for k = 1 : n3
                    for p = 1 : n2
                        dw1(q,r)=dw1(q,r)-((ytrain(k,i)-a3(k))*z3(k)*(1-z3(k))*w3(k,p)*z2(p)*(1-z2(p))*w2(p,q)*z1(q)*(1-z1(q))*xtrain(i,r));
                    end
                end
            end
        end
    end
    J(t) = j/(2);
    b3 = b3 - alpha*db3;w3 = w3 - alpha*dw3;
    b2 = b2 - alpha*db2;w2 = w2 - alpha*dw2;
    b1 = b1 - alpha*db1;w1 = w1 - alpha*dw1;
end

% TESTING
accuracy = 0;
for i = 1 : 644
    for q = 1 : n1
        for r = 1 : n0
            z1(q) = z1(q) + w1(q,r)*xtest(i, r);
        end
    z1(q) = z1(q) + b1(q);
    end
    a1 = sigmoidFunction(z1);
    for p = 1 : n2
        for q = 1 : n1
            z2(p) = z2(p) + w2(p,q)*a1(q);
        end
        z2(p) = z2(p) + b2(p);
    end
    a2 = sigmoidFunction(z2);
    for k = 1 : n3
        for p = 1 : n2
            z3(k) = z3(k) + w3(k,p)*a2(p);
        end
        z3(k) = z3(k) + b3(k);
    end
    a3 = sigmoidFunction(z3);
    [prob, pred] = max(a3);
    if pred == ytest(i)
        accuracy = accuracy + 1;
    end
end
accuracy = (100*accuracy)/644;

function g = sigmoidFunction(z)
%	Compute sigmoidFunction function

  % You need to return the following variables correctly
  g = zeros(size(z));

  % Instructions: z can be a matrix, vector or scalar
  g = 1.0 ./ ( 1.0 + exp(-z)); % For Matlab
  % g = 1.0 ./ ( 1.0 + e.^(-z)); % For Octave, it can use 'exp(1)' or 'e'
    
end