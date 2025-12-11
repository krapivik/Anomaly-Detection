clear
close all
% run("model_architecture.m")
% run("model_training.m")
load("Autoencoder.mat")

numOutputs = 1;
miniBatchSize = 32;
mbqTest = minibatchqueue(imds,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

mbqTest.shuffle

X=next(mbqTest);
Z = predict(netE,X);
Y=predict(netD,Z);
relu_1=minibatchpredict(netE,X,'Outputs','relu_1');
relu_2=minibatchpredict(netE,X,'Outputs','relu_2');

figure
t_1=tiledlayout(4,4);
title(t_1,"Активации relu_1")
for i = 1:16
    nexttile
    imshow(relu_1(:,:,i,1).extractdata)
end

figure
t_2=tiledlayout(4,8);
title(t_2,"Активации relu_2")
for i = 1:32
    nexttile
    imshow(relu_2(:,:,i,1).extractdata)
end

figure
t=tiledlayout(8,8);
title(t,"Реконструированные изображения")
for i=1:miniBatchSize
    nexttile
    imshow(X(:,:,:,i).extractdata)
    nexttile
    imshow(Y(:,:,:,i).extractdata)
end

err=abs(Y-X);
err=err.extractdata;
figure
histogram(err)
title("Абсолютная ошибка")


% clear('X','Y','Z')
% imds_2=imdsTest.subset(imdsTest.Labels=='2');
% 
% idx=randi(200,1);
% img=single(imds_2.readimage(idx));
% X=img;
% Z = predict(netE,X);
% Y=predict(netD,Z);
% 
% figure
% tiledlayout(1,2)
% nexttile
% imshow(X)
% nexttile
% imshow(Y)