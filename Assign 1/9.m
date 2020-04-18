X = xlsread('data4.xlsx');
X(:,1:7) = (X(:,1:7)-mean(X(:,1:7)))./std(X(:,1:7)); %Input normalization
p = randperm(size(X,1)); %Shuffling data
k = 600; %Number of iterations
alpha = 0.0001; %Learning rate
K = 5; %5-fold
w1 = rand(K,8); %initial weights for 1 vs all
w2 = rand(K,8); %initial weights for 2 vs all
w3 = rand(K,8); %initial weights for 3 vs all
IA = zeros(K,3); %individual accuracies for 5 folds
OA = zeros(K,1); %overall accuracies for 5 folds
for a=1:K
train_data = [];
%Dividing data using 5-fold approach
for i=1:K
if i+a~=K+1
train_data = [train_data; [ones(0.2*size(X,1),1) X(p(0.2*(i-1)*size(X,1)+1:0.2*i*size(X,1)),:)]];

else
test_data = [ones(0.2*size(X,1),1) X(p(0.2*(i-1)*size(X,1)+1:0.2*i*size(X,1)),:)];

end
end
% 1 vs all classifier
t = train_data;
for i = 1:size(train_data,1)
if train_data(i,9)==1
t(i,9)=0;
else
t(i,9)=1;
end
end
y = t(:,9); %target outputs
for i=1:k
g = logsig(w1(a,:)*t(:,1:8)')'; %hypothesis calculation
%weight updates
for j = 1:size(t,2)-1
w1(a,j) = w1(a,j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j));
end
end
clear y g t i j
% 2 vs all classifier
t = train_data;
for i = 1:size(train_data,1)
if train_data(i,9)==2
t(i,9)=0;
else
t(i,9)=1;
end
end
y = t(:,9); %target outputs
for i=1:k
g = logsig(w2(a,:)*t(:,1:8)')'; %hypothesis calculation
%weight updates
for j = 1:size(t,2)-1
w2(a,j) = w2(a,j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j));
end
end
clear y g t i j
% 3 vs all classifier
t = train_data;
for i = 1:size(train_data,1)
if train_data(i,9)==3
t(i,9)=0;
else
t(i,9)=1;

end
end
y = t(:,9); %target outputs
for i=1:k
g = logsig(w3(a,:)*t(:,1:8)')'; %hypothesis calculation
%weight updates
for j = 1:size(t,2)-1
w3(a,j) = w3(a,j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j));
end
end
clear y g t i j
%Testing using testdata
td = test_data;
yp_1 = logsig(w1(a,:)*td(:,1:8)')'; %prediction of 1 vs 2 and 3
yp_2 = logsig(w2(a,:)*td(:,1:8)')'; %prediction of 2 vs 1 and 3
yp_3 = logsig(w3(a,:)*td(:,1:8)')'; %prediction of 3 vs 1 and 2
u = [yp_1 yp_2 yp_3];
yp = zeros(size(yp_1,1),1);
for i = 1:size(yp_1,1)
%class is determined by the maximum prediction
[~, yp(i)] = max(u(i,:));
end
%Calculating confusion matrix
c = zeros(3);
for i = 1:size(yp,1)
if test_data(i,9) == 1
c(1,yp(i)) = c(1,yp(i))+1;
elseif test_data(i,9) == 2
c(2,yp(i)) = c(2,yp(i))+1;
else
c(3,yp(i)) = c(3,yp(i))+1;
end
end
%Calculating individual and overall accuracies
for i = 1:3
IA(a,i) = c(i,i)/sum(c(i,:));
OA(a) = OA(a) + c(i,i);
end
OA(a) = OA(a)/sum(c(:));
end
OverallAccuracy = mean(OA) %Calculating overall accuracy
IA
OA
