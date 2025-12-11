close all
clear 

inputSize=[28 28 1];

% numLatentChannels=12544;
% numLatentChannels=3136;
layersE=[
    imageInputLayer(inputSize,"Normalization","none")
    convolution2dLayer(5,16,"Padding","same","WeightL2Factor",100) 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,32,"Padding","same","WeightL2Factor",100) 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,64,"Padding","same","WeightL2Factor",100) 
    batchNormalizationLayer
    reluLayer

    flattenLayer()
    fullyConnectedLayer(196)
    ];

numInputChannels = inputSize(3);

layersD=[
    featureInputLayer(196)
    fullyConnectedLayer(3136)

    reshapeLayer(7,7,[],"OperationDimension","spatial-channel")

    transposedConv2dLayer(3,64,"Stride",1,"Cropping","same")
    batchNormalizationLayer
    reluLayer

    transposedConv2dLayer(5,32,"Stride",2,"Cropping","same")
    reluLayer

    transposedConv2dLayer(5,16,"Stride",2,"Cropping","same")
    reluLayer
    transposedConv2dLayer(5,numInputChannels, "Cropping","same")

    sigmoidLayer
    ];

netE = dlnetwork(layersE);
netD = dlnetwork(layersD);
% analyzeNetwork(netE);
% analyzeNetwork(netD);
save("Architecture.mat","netE","netD");
