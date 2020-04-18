data=importdata('data5.mat');
X=data(:,1:end-1);
Y=data(:,end);
s = RandStream('mt19937ar','Seed',1);
trainInput=[]; %Define arrays to segregate training and testing data
trainOutput=[];
testInput=[];
testOutput=[];
for j=1:size(X,1)
    if rand<0.7
        trainInput=[trainInput;X(j,:)];
        trainOutput=[trainOutput;Y(j,:)];
    else
        testInput=[testInput;X(j,:)];
        testOutput=[testOutput;Y(j,:)];
    end
end
x=trainInput;
y=trainOutput;
xt=testInput;
yt=testOutput;
[l,mu]=kmeans(x,10);
H=zeros(1495,10);
for i=1:size(x,1)
    for j=1:size(mu,1)
        H(i,j)=(norm(x(i,:)-mu(j,:)))^3; %Cubic kernel
    end
end
kk=pinv(H); %Pseudoinverse of matrix
w=kk*y;
for i1=1:size(xt,1)
    for j=1:size(mu,1)
        Ht(i1,j)=(norm(xt(i1,:)-mu(j,:)))^3;
    end
end
yp=Ht*w;
yp(yp>0.5) = 1;
yp(yp<0.5) = 0;
[cm, order]=confusionmat(yt,yp);
IA_1 = cm(1,1)/(cm(1,1) + cm(1,2));
IA_2 = cm(2,2)/(cm(2,1) + cm(2,2));
OA = (cm(1,1) + cm(2,2))/(sum(sum(cm)));