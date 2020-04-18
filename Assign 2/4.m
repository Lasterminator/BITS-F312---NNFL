clc;
clear;
data=importdata('data5.mat');
% normalization of the data
data(:,1:72)= (data(:,1:72)-mean(data(:,1:72)));
data(:,1:72)= (data(:,1:72)./std(data(:,1:72)));
% divide data into 70:30
data1=data(randperm(size(data,1)),:);
datatr=data1(1:1504,:);
datatst=data1(1505:2148,:);
% ytst:test output
ytst=datatst(:,73);
% ytr:training output
ytr=datatr(:,73);
datatr=[ones(size(datatr,1),1),datatr(:,1:72)];
datatst=[ones(size(datatst,1),1),datatst(:,1:72)];
%training with tanh activation function
n=input('Enter the no. of hidden neurons: ');
win=randn(size(datatr,2),n);
temp=datatr*win;
h=tanh(temp);
wout=pinv(h)*ytr;
%testing with tanh activation function
temp1=datatst*win;
h1=tanh(temp1);
ypr=h1*wout;
%class allocation, ypr:final output(predicted)
for i=1:size(ypr,1)
    if(ypr(i)>0.5)
        ypr(i)=1;
    else
        ypr(i)=0;
    end
end
% accuracy calculation
t1=0;
for i=1:size(ypr,1)
    if(ypr(i)==ytst(i))
        t1=t1+1;
    end
end
% gives confusion matrix
[cm,a]=confusionmat(ytst,ypr);
cm
accuracy = t1/644
