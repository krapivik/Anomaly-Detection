clear 

dataFolder="D:\Rozhkova\Ваня\DL\Ex_5\mvtec_anomaly_detection\transistor\train\good";
imdsTrain = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTrain.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));
inputSize=[128 128 1];
numLatentChannels=512;
layersE=[
    imageInputLayer(inputSize,"Normalization","none")

    convolution2dLayer(3,16,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer("Name",'activation_1_1')
    % convolution2dLayer(3,16,"Padding","same") 
    % leakyReluLayer("Name",'activation_1_2')

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,32,"Padding","same") 
    batchNormalizationLayer
    leakyReluLayer("Name",'activation_2_1')
    % convolution2dLayer(3,32,"Padding","same") 
    % leakyReluLayer("Name",'activation_2_2')

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,64,"Padding","same")     
    leakyReluLayer("Name",'activation_3_1')
 
    % convolution2dLayer(3,64,"Padding","same","WeightL2Factor",1) 
    % reluLayer    

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,128,"Padding","same") 
    leakyReluLayer("Name",'activation_4')
    % convolution2dLayer(3,128,"Padding","same","WeightL2Factor",1) 
    % reluLayer

    maxPooling2dLayer(2,"Stride",2)
    
    convolution2dLayer(3,256,"Padding","same") 
    leakyReluLayer("Name",'activation_5')

    maxPooling2dLayer(2,"Stride",2)
    flattenLayer

    fullyConnectedLayer(numLatentChannels)
    ];

numInputChannels = inputSize(3);

layersD=[
    featureInputLayer(numLatentChannels)
    fullyConnectedLayer(4096)

    reshapeLayer(4,4,[],"OperationDimension","spatial-channel")

    transposedConv2dLayer(3,256,"Stride",2,"Cropping","same")
    reluLayer

    transposedConv2dLayer(3,128,"Stride",2,"Cropping","same")
    reluLayer

    transposedConv2dLayer(3,64,"Stride",2,"Cropping","same")
    reluLayer
    
    transposedConv2dLayer(3,32,"Stride",2,"Cropping","same")
    reluLayer

    transposedConv2dLayer(3,16,"Stride",2,"Cropping","same")
    reluLayer
    transposedConv2dLayer(3,numInputChannels, "Cropping","same")

    sigmoidLayer
    ];

netE = dlnetwork(layersE);
netD = dlnetwork(layersD);
% analyzeNetwork(netE);
% analyzeNetwork(netD);
save("Architecture_3.mat","netE","netD");
