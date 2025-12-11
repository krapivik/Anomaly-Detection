clear
close all
load("Architecture.mat")

%--------------------------------------------------------------------------
% dataFolder="D:\Rozhkova\Ваня\DL\data\Concrete Crack Images for Classification";
% imds = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
%
% imds_neg=imds.subset(imds.Labels=='Negative');
% [imdsTrain,imdsTest]=splitEachLabel(imds_neg,0.8,'randomized');
% inputSize=size(imds.readimage(1));
% imdsTrain=subset(imdsTrain,1:5000);
% 
%--------------------------------------------------------------------------

dataFolder="D:\Rozhkova\Ваня\DL\data\DigitsData";
imds = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imds.ReadFcn = @(filename) im2single(imread(filename)); %???
imds_neg=imds.subset(imds.Labels=='2');
[imdsTrain,imdsTest]=splitEachLabel(imds,0.8,'randomized');
inputSize=[28 28 1];

%--------------------------------------------------------------------------
numEpochs = 10;
miniBatchSize = 32;
learnRate = 1e-3;

trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];

numObservationsTrain = numel(imdsTrain.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

numOutputs=1;
mbq=minibatchqueue(imdsTrain,numOutputs, ...
    'MiniBatchSize',miniBatchSize,...
    "MiniBatchFcn",@preprocessMiniBatch,...
    MiniBatchFormat = 'SSCB', ...
    PartialMiniBatch = 'discard');

monitor = trainingProgressMonitor(Metrics="Loss", ...
    Info=["Epoch" "LearnRate"],...
    Xlabel="Iteration", ...
    Visible='on');

%------------------------------------------------------------------------

epoch = 0;
iteration = 0;
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(mbq)
    while mbq.hasdata && ~monitor.Stop
        iteration = iteration + 1;
        X = next(mbq);

        % Evaluate loss and gradients.
        [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,X);

        % Update learnable parameters.
        [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
            gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);

        % Update the training progress monitor. 
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end
end

save("Autoencoder.mat","netE","netD","imds","imdsTest");