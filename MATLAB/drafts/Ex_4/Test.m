clear

addpath 'D:\Rozhkova\Ваня\DL\Ex_4\Activation_Measure'
addpath 'D:\Rozhkova\Ваня\DL\Ex_4'\Error_Measure\
% run("model_architecture.m")
% run("model_training.m")
load("Autoencoder.mat")

numOutputs = 1;
miniBatchSize = 32;
mbqTest = minibatchqueue(imdsTest,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

mbqTest.shuffle

X=next(mbqTest);
Z = predict(netE,X);
Y=predict(netD,Z);
relu_1=minibatchpredict(netE,X,'Outputs','relu_1');
relu_2=minibatchpredict(netE,X,'Outputs','relu_2');

%%

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

err=[];
while mbqTest.hasdata
    next(mbqTest);
    Z=predict(netE,X); Y=predict(netD,Z);
    for i=1:size(X,3)
        for j=1:size(X,4)
        err = [err mean(abs(Y(:,:,i,j)-X(:,:,i,j)),"all")];
        end
    end
end
err=err.extractdata;
figure
histogram(err)

[back_err,front_err]=BF_relative_error(X,Y); [back_err front_err]

% clear('X','Y','Z')
% X=next();
% Z=predict(netE,X);
% Y=predict(netD,Z);
% err=abs(Y-X);
% err=err.extractdata;
% figure
% histogram(err)
% title("Абсолютная ошибка")


%%
% A_1=mean_act(relu_1);
% A_2=sparse_act(relu_1,0);
% 
% 
% clear('X','Y','Z','relu_1','relu_2')
% imds_2=imdsTest.subset(imdsTest.Labels=='2');
% relu_1=minibatchpredict(netE,imds_2,'Outputs','relu_1');
% relu_2=minibatchpredict(netE,imds_2,'Outputs','relu_2');
% 
% act_1=mean(mean_act(relu_2),2)*100;
% 
% clear('X','Y','Z','relu_1','relu_2','imds_2')
% imds_2=imdsTest.subset(imdsTest.Labels=='4');
% relu_1=minibatchpredict(netE,imds_2,'Outputs','relu_1');
% relu_2=minibatchpredict(netE,imds_2,'Outputs','relu_2');
% 
% act_2=mean(mean_act(relu_2),2)*100;
% 
% [act_1 act_2]
% idx=randi(200,1);
% img=single(imds_2.readimage(idx));
% X=img;
% Z = predict(netE,X);
% Y=predict(netD,Z);

% figure
% tiledlayout(1,2)
% nexttile
% imshow(X)
% nexttile
% imshow(Y)