
% Agriculture Yield Prediction using Random Forest

data = readtable('../data/raw/agriculture_dataset.csv');

X = data{:,1:end-1};
y = data.crop_yield;

cv = cvpartition(size(X,1),'HoldOut',0.2);
idx = cv.test;

XTrain = X(~idx,:);
yTrain = y(~idx);
XTest = X(idx,:);
yTest = y(idx);

model = TreeBagger(300,XTrain,yTrain,'Method','regression');

pred = predict(model,XTest);

mse = mean((pred - yTest).^2);
disp("MSE:");
disp(mse);
