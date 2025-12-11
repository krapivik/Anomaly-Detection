clear

addpath 'D:\Rozhkova\Ваня\DL\Ex_4\Activation_Measure'
addpath 'D:\Rozhkova\Ваня\DL\Ex_4'\Error_Measure\
run("model_architecture_2.m")
run("model_training_2.m")
load("Autoencoder_2.mat")

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
relu_3=minibatchpredict(netE,X,'Outputs','relu_3');
relu_4=minibatchpredict(netE,X,'Outputs','relu_4');
%%

% figure
% t_1=tiledlayout(4,4);
% title(t_1,"Активации relu_1")
% for i = 1:16
%     nexttile
%     imshow(relu_1(:,:,i,1).extractdata)
% end
% 
% figure
% t_2=tiledlayout(4,4);
% title(t_2,"Активации relu_2")
% for i = 1:16
%     nexttile
%     imshow(relu_2(:,:,i,1).extractdata)
% end
% 
% figure
% t_3=tiledlayout(8,4);
% title(t_3,"Активации relu_3")
% for i = 1:32
%     nexttile
%     imshow(relu_3(:,:,i,1).extractdata)
% end
% 
% figure
% t_4=tiledlayout(8,4);
% title(t_4,"Активации relu_4")
% for i = 1:32
%     nexttile
%     imshow(relu_4(:,:,i,1).extractdata)
% end

%%
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

[back_err,front_err]=BF_relative_error(X,Y); %Перепроверить функцию
[back_err front_err]
[mean(err) std(err)]


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