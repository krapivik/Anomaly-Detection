clear

addpath 'D:\Rozhkova\Projects\DL\functions\'

run("model_architecture.m")
run("model_training.m")
% close all
load("Autoencoder.mat")

dataFolder="D:\Rozhkova\Ваня\DL\data\mvtec_anomaly_detection\screw\test\good";
imdsTest = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTest.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));
inputSize=[128 128 1];

augimds = augmentedImageDatastore([128 128], imdsTest);

numOutputs = 1;
miniBatchSize = 12;
mbqTest = minibatchqueue(augimds,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

mbqTest.shuffle

X=next(mbqTest);
Z = predict(netE,X);
Y=predict(netD,Z);

%%

featureMap(netE,X,'E activation_1_1');
featureMap(netE,X,'E activation_2_1');
featureMap(netE,X,'E activation_3_1');
featureMap(netE,X,'E activation_4_1');
% 
featureMap(netD,Z,'D activation_1_1');
featureMap(netD,Z,'D activation_2_1');
featureMap(netD,Z,'D activation_3_1');
featureMap(netD,Z,'D activation_4_1');
featureMap(netD,Z,'D activation_5_1');
% 
figure
t=tiledlayout(6,4);
title(t,"Реконструированные изображения")
for i=1:miniBatchSize
    nexttile
    imshow(X(:,:,:,i).extractdata)
    nexttile
    imshow(Y(:,:,:,i).extractdata)
end

figure
t=tiledlayout(4,6);
title(t,'Распределение значений вектора свертки')
for i=1:miniBatchSize
    nexttile
    imshow(X(:,:,:,i).extractdata)
    nexttile
    histogram(Z(:,i).extractdata,30,'Orientation','vertical','BinLimits', [-0.5 0.5])
    mu=mean(Z(:,i).extractdata);
    sigma=std(Z(:,i).extractdata);
    subtitle([strcat("Mean: ", num2str(mu)) ...
        strcat("STD: ",num2str(sigma))])
end

%%

err=[];
max_err=[];
while mbqTest.hasdata
    next(mbqTest); % переход к следующему батчу
    Z=predict(netE,X); Y=predict(netD,Z);
    for i=1:size(X,3) % номер канала изображения
        for j=1:size(X,4) % номер изображения в батче      
        err = [err mean(abs(Y(:,:,i,j)-X(:,:,i,j)),"all")]; % Средняя ошибка одного изображения
        max_err = [max_err max(Y(:,:,i,j)-X(:,:,i,j),[],"all")];
        end
    end
end

err=err.extractdata;
max_err=max_err.extractdata;
figure
tiledlayout(1,2)
nexttile
histogram(err)
title("Среднее значение ошибки тестовых данных")
nexttile
histogram(max_err)
title("Максимальные значения ошибки тестовых данных")

% [mean(err) std(err)]

errorMap(X(:,:,:,1).extractdata,Y(:,:,:,1).extractdata);


a = -0.5;
b = 0.5;
n = 128;

randZ = dlarray(a + (b-a).*rand(n,12),'CB');
randY=predict(netD,randZ);

figure
t=tiledlayout(4,6);
title(t,"Из случайного вектора равномерного распределения")
for i=1:miniBatchSize
    nexttile
    imshow(randY(:,:,:,i).extractdata)
    nexttile
    histogram(randZ(:,i).extractdata, 30,'BinLimits', [-0.5 0.5])
    mu=mean(Z(:,i).extractdata);
    sigma=std(Z(:,i).extractdata);
    subtitle([strcat("Mean: ", num2str(mu)) ...
        strcat("STD: ",num2str(sigma))])
end

a = 0.15;
b = 0;
randZ = dlarray(a.*randn(128,12) + b,"CB");
randY=predict(netD,randZ);

figure
t=tiledlayout(4,6);
title(t,"Из случайного вектора нормального распределения")
for i=1:miniBatchSize
    nexttile
    imshow(randY(:,:,:,i).extractdata)
    nexttile
    histogram(randZ(:,i).extractdata,30,'BinLimits', [-0.5 0.5])
       mu=mean(Z(:,i).extractdata);
    sigma=std(Z(:,i).extractdata);
    subtitle([strcat("Mean: ", num2str(mu)) ...
        strcat("STD: ",num2str(sigma))])
end