
clear 

inputSize=[28 28 1];
numLatentChannels=196;
layersE=[
    imageInputLayer(inputSize,"Normalization","none")
    convolution2dLayer(5,16,"Padding","same","WeightL2Factor",50) 
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(5,32,"Padding","same","WeightL2Factor",50) 
    batchNormalizationLayer
    reluLayer

    % maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,64,"Padding","same","WeightL2Factor",50) 
    batchNormalizationLayer
    reluLayer

    flattenLayer()
    fullyConnectedLayer(numLatentChannels)
    ];

numInputChannels = inputSize(3);

layersD=[
    featureInputLayer(numLatentChannels)
    fullyConnectedLayer(3136)

    reshapeLayer(14,14,[],"OperationDimension","spatial-channel")

    transposedConv2dLayer(3,64,"Stride",1,"Cropping","same")
    batchNormalizationLayer
    reluLayer

    transposedConv2dLayer(5,32,"Stride",1,"Cropping","same")
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
