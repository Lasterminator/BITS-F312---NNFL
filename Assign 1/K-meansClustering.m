data = xlsread('data2.xlsx');
iter_nos = 100;
cluster_nos = 2;
[c_nos, c] = kmeans(data, cluster_nos, iter_nos);
figure;
scatter(data(:, 1), c);                  %Class vs Feature 
hold on;
scatter(c_nos(:, 1), 1:2);
figure;
scatter(data(:, 2), c);                  %Class vs Feature 2
hold on;
scatter(c_nos(:, 2), 1:2);
figure;
scatter(data(:, 3), c);
hold on;
scatter(c_nos(:, 3), 1:2);               %Class vs Feaure 3
figure;
scatter(data(:, 4), c);
hold on;
scatter(c_nos(:, 4), 1:2);               % Class vs Feature 4


function [c_center, c] = kmeans(X, cluster_nos, iter_nos)
    [rows, cols] = size(X);
    c_center = zeros(cluster_nos, cols);              %Cluster centers
    c = zeros(rows, 1);                               %Array to assign elemnets to each cluster center
    data_num = zeros(cluster_nos, 1);                 %Array to store elements of the clusters
    X_new = randperm(size(X, 1), cluster_nos);
    for i = 1:1:cluster_nos
        c_center(i,:) = X(X_new(i), :);
    end
    for i = 1:1:iter_nos
        dist = zeros(rows, cluster_nos);
        for j = 1:1:rows
            for k = 1:1:cluster_nos
                dist(j, k) = sqrt(sum((X(j, :) - c_center(k, :)).^2));      %Finding distance between clusters and data points
            end
            c(j) = find(dist(j,: ) == min(dist(j,: )),1);
        end
        for k = 1:1:cluster_nos
            data_num(k) = sum(c(:) == k);
            c_center(k,:) = sum(X(find(c(:) == k),:))/data_num(k);          %finding mean of clusters to recalculate cluster centers
        end
    end
end
