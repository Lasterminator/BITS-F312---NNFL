clc;
clear;
data=importdata('data5.mat');
% normalization of the data
data(:,1:72)= (data(:,1:72)-mean(data(:,1:72)));
data(:,1:72)= (data(:,1:72)./std(data(:,1:72)));
% divide data into 70:30
data1=data(randperm(size(data,1)),:);
datatr=(data1(1:1504,:));
datatst=(data1(1505:2148,:));
% ytst:test output
ytst=(datatst(:,73));
% ytr:trainning output
ytr=(datatr(:,73));
% converting to rows
datatr=(data1(1:1504,:))';
datatst=(data1(1505:2148,:))';
%for training data class allocation
for i=1:size(ytr)
    if(ytr(i)==1)
        ymat(i,2)=1;
        ymay(i,1)=0;
    else
        ymat(i,1)=1;
        ymat(i,2)=0;
    end
end
ymat=ymat';
%adding baise values to the feature matrix
datatr=[ones(1,size(datatr,2));datatr];
datatst=[ones(1,size(datatst,2));datatst];
%first encoder input is training data
%Maximum number of training epochs or iterations, specified as the
%comma-separated pair consisting of 'MaxEpochs' and a positive
%integer value.
autoen1=trainAutoencoder(datatr,50,'MaxEpochs',2000);
xpr1=predict(autoen1,datatr);
feat1=encode(autoen1,datatr);
%second encoder its input is previous ones output
autoen2=trainAutoencoder(feat1,30,'MaxEpochs',2000);
feat2=encode(autoen2,feat1);
%third encoder its input is previous ones output
autoen3=trainSoftmaxLayer(feat2,ymat,'MaxEpochs',2000);
stackednet=stack(autoen1,autoen2,autoen3);
%fine tuning
stackednet=train(stackednet,datatr,ymat)
%prediction output is in fractions
yprmat=stackednet(datatst);
%class allocation, ypr:final output(predicted)
for i=1:size(yprmat,2)
    if(yprmat(1,i)>yprmat(2,i))
        ypr(i)=0;
    else
        ypr(i)=1;
    end
end
% accuracy calculation
ytst=ytst';
t=0;
for i=1:size(ypr,2)
    if(ypr(i)==ytst(i))
        t=t+1;
    end
end
accuracy=t/644