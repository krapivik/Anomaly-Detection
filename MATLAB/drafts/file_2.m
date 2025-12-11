clear 
close all

unzip("DigitsData.zip");
dataFolder = "DigitsData";
imds = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");

imds_new=imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");

imds_new.Labels=categorical(imds_new.Labels=='2'); % Переопределение меток класса
imds_True=subset(imds_new,imds_new.Labels=='true'); % Набор изображений только с меткой "true"


[imdsTrain, imdsTest] = splitEachLabel(imds_True, 0.8, 'randomized');
im_train=imdsTrain.readall;
im_test=imdsTest.readall;
hiddenSize_1 = 100;
autoenc_1 = trainAutoencoder(im_train,hiddenSize_1,...
    'MaxEpochs',400,...
    'L2WeightRegularization',0.004,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.15,...
    'ScaleData',true);


% imds_test=subset(imds,imds.Labels=='2');
% imtest_2=imds_test.readall;
% 
% 
% XReconstructed  = predict(autoenc_1,imtest_2);
% 
% figure
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imtest_2{i});
% end
% 
% figure
% for i = 1:20
%     subplot(4,5,i);
%     imshow(XReconstructed {i});
% end
% 
% for i =1:size(XReconstructed,2)
% % img_err(i)=mean((double(imtest_2{i})-XReconstructed{i}).^2,'all');
% img_err(i)=mse(double(imtest_2{i})-XReconstructed{i},'all');
% end
% 
% figure
% histogram(img_err)
