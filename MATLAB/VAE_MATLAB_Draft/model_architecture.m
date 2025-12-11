clear 

dataFolder="D:\Rozhkova\Projects\DL\data\mvtec_anomaly_detection\transistor\train\good";
imdsTrain = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTrain.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));
inputSize=[128 128 1];
numLatentChannels=128;

numInputChannels = inputSize(3);
layersE=[
    imageInputLayer(inputSize,"Normalization","none")

    convolution2dLayer(3,8,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer("Name",'E activation_1_1')

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,16,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer("Name",'E activation_2_1')
 
    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,32,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer("Name",'E activation_3_1')

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,64,"Padding","same") 
    batchNormalizationLayer
    leakyReluLayer("Name",'E activation_4_1')

    maxPooling2dLayer(2,"Stride",2)
    
    convolution2dLayer(3,128,"Padding","same") 
    batchNormalizationLayer
    leakyReluLayer("Name",'E activation_5_1')

    maxPooling2dLayer(2,'Name','Pool',"Stride",2)

    % flattenLayer('Name','Flatten')
    ];

layersD=[    
    fullyConnectedLayer(2048,'Name',"fcDecoder")
    reshapeLayer(4,4,[],"OperationDimension","spatial-channel")

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,128,"Padding","same")
    leakyReluLayer("Name",'D activation_1_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,64,"Padding","same")
    leakyReluLayer("Name",'D activation_2_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,32,"Padding","same")
    leakyReluLayer("Name",'D activation_3_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,16,"Padding","same","WeightsInitializer","he")
    leakyReluLayer("Name",'D activation_4_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,8,"Padding","same")
    leakyReluLayer("Name",'D activation_5_1')

    transposedConv2dLayer(3,numInputChannels, "Cropping","same")

    sigmoidLayer];

mu=fullyConnectedLayer(numLatentChannels,"Name","mean");
logSigmaSq=fullyConnectedLayer(numLatentChannels,"Name","log-variance");

% Слои перед сэмплированием
net = dlnetwork(layersE);
net=addLayers(net,mu);
net=connectLayers(net,"Pool","mean");
net=addLayers(net,logSigmaSq);
net=connectLayers(net,"Pool","log-variance");

sampling=samplingLayer('Name',"Sampling");
net=addLayers(net,sampling);
net=connectLayers(net,"mean","Sampling/mu"); 
net=connectLayers(net,"log-variance","Sampling/logSigmaSq");



net=addLayers(net,layersD);
net=connectLayers(net,"Sampling","fcDecoder");
net=net.initialize;

analyzeNetwork(net);


netD=dlnetwork
netD=addlayers(netD,layersD)
analyzeNetwork(netD);
save("Architecture.mat","net");


