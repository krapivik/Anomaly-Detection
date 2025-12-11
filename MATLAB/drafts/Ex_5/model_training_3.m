clear
load("Architecture_3.mat")

%--------------------------------------------------------------------------

dataFolder="D:\Rozhkova\Ваня\DL\Ex_5\mvtec_anomaly_detection\screw\train\good";
imdsTrain = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTrain.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));


inputSize=[128 128 1];

augimds = augmentedImageDatastore([128 128], imdsTrain);
%--------------------------------------------------------------------------
numEpochs = 50;
miniBatchSize = 64;
learnRate = 1e-3;

trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];

numObservationsTrain = numel(augimds.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

numOutputs=1;
mbq=minibatchqueue(augimds,numOutputs, ...
    'MiniBatchSize',miniBatchSize,...
    "MiniBatchFcn",@preprocessMiniBatch,...
    MiniBatchFormat = 'SSCB', ...
    PartialMiniBatch = 'discard');

monitor = trainingProgressMonitor(Metrics="Loss", ...
    Info=["Epoch" "LearnRate"],...
    Xlabel="Iteration", ...
    Visible='on');

%------------------------------------------------------------------------
grad_trace=[];
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
        % grad_trace=[grad_trace;gradNorm(gradientsE)];r

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

% figure 
% tiledlayout(2,1)
% plot(grad_trace(:,[1 5 7 9 11 13])) 
% legend
% plot(grad_trace(:,[2 6 8 10 12 14])) 
% legend

save("Autoencoder_3.mat","netE","netD");