% Здесь пробуем писать CNN
clear 
close all


unzip("DigitsData.zip");
dataFolder = "DigitsData";
imds = imageDatastore(dataFolder, ...
 IncludeSubfolders=true, ...
 LabelSource="foldernames");

idx=imds.Labels=='2';

imds.Labels=categorical(imds.Labels=='2'); % Переопределение меток класса
imds.Labels;
imds_True=subset(imds,imds.Labels=='true'); % Набор данных с меткой "true"

net = dlnetwork; % Инициализация нейросети

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized'); 

[h,w]=size(readimage(imds,1)); 
numChannels=1;
numObservations=length(imds.Labels);

imageInputSize = [h w numChannels];
filterSize = 5; numFilters = 16; % Параметры свёрточного слоя
layers=[imageInputLayer(imageInputSize,"Normalization","none")
    convolution2dLayer(3,10,"Padding","same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2) % Полносвязный слой для бинарной классификации (2 класса)
    softmaxLayer 
    ];
net=addLayers(net,layers);

figure
plot(net)

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ... % Количество эпох
    'MiniBatchSize', 128, ... % Размер мини-пакета
    'Shuffle', 'every-epoch', ... % Перемешивание данных
    'ValidationFrequency', 30, ... % Частота проверки на валидации
    'Verbose', 1, ... % Вывод прогресса
    'Plots', 'training-progress'); % График обучения

netTrained = trainnet(imdsTrain, net, "crossentropy", options);

XTest=readall(imdsTest); 
TTest=imdsTest.Labels;
classNames=categories(TTest);
XTest=cat(4,XTest{:}); XTest=single(XTest);

YTest = minibatchpredict(netTrained,XTest);
YTest = onehotdecode(YTest,classNames,2);

figure
confusionchart(TTest,YTest,"Normalization","row-normalized")

% Пример классификации одного изображения
img = readimage(imdsTest, 1); % Берем первое тестовое изображение
label = imdsTest.Labels(1); % Истинная метка
pred = minibatchpredict(netTrained, img); % Предсказание
% figure 
% imshow(img)
% pred
% label



