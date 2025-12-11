clear 

dataFolder="D:\Rozhkova\Projects\DL\data\mvtec_anomaly_detection\transistor\train\good";
imdsTrain = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTrain.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));
inputSize=[128 128 1];
numLatentChannels=128;
layersE=[
    imageInputLayer(inputSize,"Normalization","none")

    convolution2dLayer(3,8,"Padding","same","PaddingValue","symmetric-exclude-edge", ...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'E activation_1_1')

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,16,"Padding","same","PaddingValue","symmetric-exclude-edge", ...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'E activation_2_1')

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,32,"Padding","same","PaddingValue","symmetric-exclude-edge", ...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'E activation_3_1')


    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,64,"Padding","same","PaddingValue","symmetric-exclude-edge", ...
    "WeightsInitializer","he") 
    leakyReluLayer("Name",'E activation_4_1')

    maxPooling2dLayer(2,"Stride",2)
    
    convolution2dLayer(3,128,"Padding","same","PaddingValue","symmetric-exclude-edge", ...
    "WeightsInitializer","he") 
    leakyReluLayer("Name",'E activation_5_1')

    maxPooling2dLayer(2,"Stride",2)
    flattenLayer

    fullyConnectedLayer(numLatentChannels)
    ];

netE=dlnetwork;
netE=addLayers(netE,layersE);

analyzeNetwork(netE)
netE=initialize(netE);


numInputChannels = inputSize(3);

layersD=[
    featureInputLayer(numLatentChannels)
    fullyConnectedLayer(2048)

    reshapeLayer(4,4,[],"OperationDimension","spatial-channel")

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,128,"Padding","same","PaddingValue","symmetric-exclude-edge",...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'D activation_1_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,64,"Padding","same",...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'D activation_2_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,32,"Padding","same","PaddingValue","symmetric-exclude-edge",...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'D activation_3_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,16,"Padding","same","PaddingValue","symmetric-exclude-edge",...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'D activation_4_1')

    resize2dLayer("Scale",2,"Method","bilinear")
    convolution2dLayer(3,8,"Padding","same","PaddingValue","symmetric-exclude-edge",...
    "WeightsInitializer","he")
    leakyReluLayer("Name",'D activation_5_1')

    transposedConv2dLayer(3,numInputChannels, "Cropping","same")

    sigmoidLayer
    ];

netD = dlnetwork(layersD);
analyzeNetwork(netD);

save("Architecture.mat","netE","netD");
