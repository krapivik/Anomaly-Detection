clear

addpath 'D:\Rozhkova\Ваня\DL\Ex_4\Activation_Measure'
addpath 'D:\Rozhkova\Ваня\DL\Ex_4\Error_Measure\'
% run("model_architecture_3.m")
% run("model_training_3.m")
load("Autoencoder_3.mat")

dataFolder="D:\Rozhkova\Ваня\DL\Ex_5\mvtec_anomaly_detection\screw\train\good";
imdsTest = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTest.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));
inputSize=[128 128 1];

augimds = augmentedImageDatastore([128 128], imdsTest);


numOutputs = 1;
miniBatchSize = 32;
mbqTest = minibatchqueue(augimds,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

mbqTest.shuffle

X=next(mbqTest);
Z = predict(netE,X);
Y=predict(netD,Z);

%%

% Добавить в функцию оценку интенсивности активации нейрона~~~
featureMap(netE,X,'activation_1_1');
% featureMap(netE,X,'activation_1_2');
featureMap(netE,X,'activation_2_1');
% featureMap(netE,X,'activation_2_2');
featureMap(netE,X,'activation_3_1');
featureMap(netE,X,'activation_4');

figure
t=tiledlayout(8,8);
title(t,"Реконструированные изображения")
for i=1:miniBatchSize
    nexttile
    imshow(X(:,:,:,i).extractdata)
    nexttile
    imshow(Y(:,:,:,i).extractdata)
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

% [back_err,front_err]=BF_relative_error(X,Y);
% [back_err front_err]
% [mean(err) std(err)]

errorMap(X(:,:,:,1).extractdata,Y(:,:,:,1).extractdata)


%%

% clear('X','Y','Z')
% idx=randi(numel(imds.Files),[1 32]);
% X=imds.subset(idx);
% 
% figure
% tiledlayout(8,8)
% for i = 1:32
%     Z=predict(netE,X.readimage(i)); Y=predict(netD,Z);
%     nexttile
%     imshow(X.readimage(i))
%     nexttile
%     imshow(Y)
% end


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