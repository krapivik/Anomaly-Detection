clear
close all
addpath("C:\Users\rozhkova_as\Documents\MATLAB\Examples\R2024a\nnet\GeneratingHanddrawnDigitsUsingAVariationalAutoencoderVAEExample")

% [imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
% hiddenSize_1=100;
% imTrain=imdsTrain.readall;
% % imageSize=size(imds.readimage(1));
% numClasses = numel(categories(imds.Labels));
% 
% 
% XTrain=subset(imdsTrain,imdsTrain.Labels=='Negative').readall;

[XTrain, YTrain] = digitTrain4DArrayData;
[XTest, YTest] = digitTest4DArrayData;

idx=YTrain=="2";
XTrain=XTrain(:,:,:,idx);

numLatentChannels=32;
imageSize = [28 28 1];
layersE = [
 imageInputLayer(imageSize,Normalization="none")
 convolution2dLayer(3,32,Padding="same",Stride=2)
 reluLayer
 convolution2dLayer(3,64,Padding="same",Stride=2)
 reluLayer
 fullyConnectedLayer(2*numLatentChannels)
 samplingLayer
];

projectionSize = [7 7 64];
numInputChannels = imageSize(3);
layersD = [
 featureInputLayer(numLatentChannels)
 projectAndReshapeLayer(projectionSize)
 transposedConv2dLayer(3,64,Cropping="same",Stride=2)
 reluLayer
 transposedConv2dLayer(3,32,Cropping="same",Stride=2)
 reluLayer
 transposedConv2dLayer(3,numInputChannels,Cropping="same")
 sigmoidLayer];

netE = dlnetwork(layersE);
netD = dlnetwork(layersD);

analyzeNetwork(netE)
analyzeNetwork(netD)

numEpochs = 150;
miniBatchSize = 128;
learnRate = 1e-3;

dsTrain = arrayDatastore(XTrain,IterationDimension=4);
numOutputs = 1;

mbq = minibatchqueue(dsTrain,numOutputs, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    PartialMiniBatch="discard");
size(next(mbq))

trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];

numObservationsTrain = size(XTrain,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info="Epoch", ...
    XLabel="Iteration");

epoch = 0;
iteration = 0;

% % Loop over epochs.
% while epoch < numEpochs && ~monitor.Stop
%     epoch = epoch + 1;
% 
%     % Shuffle data.
%     shuffle(mbq);
% 
%     % Loop over mini-batches.
%     while hasdata(mbq) && ~monitor.Stop
%         iteration = iteration + 1;
% 
%         % Read mini-batch of data.
%         X = next(mbq);
% 
%         % Evaluate loss and gradients.
%         [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,X);
% 
%         % Update learnable parameters.
%         [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
%             gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);
% 
%         [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
%             gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);
% 
%         % Update the training progress monitor. 
%         recordMetrics(monitor,iteration,Loss=loss);
%         updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
%         monitor.Progress = 100*iteration/numIterations;
%     end
% end
% 
% idx=YTest=="5";
% XTest=XTest(:,:,:,idx);
% 
% ZNew = predict(netE,XTest(:,:,:,1:64))';
% ZNew = dlarray(ZNew,"CB");
% 
% YNew = predict(netD,ZNew);
% YNew = extractdata(YNew);
% 
% figure
% H = imtile(XTest(:,:,1:64));
% imshow(H)
% figure
% I = imtile(YNew);
% imshow(I)
% title("Generated Images")
% 
% 
% % size(XTest(:,:,1:64))
% % size(YNew)
% err = mean((XTest(:,:,1:64)-YNew).^2,[1 2 3]);
% 
% figure
% histogram(err)
% xlabel("Error")
% ylabel("Frequency")
% title("Test Data")
