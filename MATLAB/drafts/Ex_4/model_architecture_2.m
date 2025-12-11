clear 

inputSize=[28 28 1];
numLatentChannels=98;
layersE=[
    imageInputLayer(inputSize,"Normalization","none")
    convolution2dLayer(3,16,"Padding","same","WeightL2Factor",50)
    reluLayer
    convolution2dLayer(3,16,"Padding","same","WeightL2Factor",50) 
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,32,"Padding","same","WeightL2Factor",50) 
    reluLayer
    convolution2dLayer(3,32,"Padding","same","WeightL2Factor",50)
    reluLayer

    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,64,"Padding","same","WeightL2Factor",50) 
    reluLayer
    convolution2dLayer(3,64,"Padding","same","WeightL2Factor",50) 
    reluLayer

    fullyConnectedLayer(numLatentChannels)
    ];

numInputChannels = inputSize(3);

layersD=[
    featureInputLayer(numLatentChannels)
    fullyConnectedLayer(3136)

    reshapeLayer(7,7,[],"OperationDimension","spatial-channel")

    transposedConv2dLayer(3,64,"Stride",1,"Cropping","same")
    batchNormalizationLayer
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
save("Architecture_2.mat","netE","netD");
