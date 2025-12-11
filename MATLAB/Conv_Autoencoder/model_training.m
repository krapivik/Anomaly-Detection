clear
load("Architecture.mat")
addpath 'D:\Rozhkova\Projects\DL\functions\'
addpath 'D:\Rozhkova\Projects\DL\functions\Grad_Distributions_functions\'
addpath 'D:\Rozhkova\Projects\DL\functions'\ShowLearningIMGapp\
%--------------------------------------------------------------------------

dataFolder="D:\Rozhkova\Projects\DL\data\mvtec_anomaly_detection\screw\train\good";
imdsTrain = imageDatastore(dataFolder,IncludeSubfolders=true, LabelSource="foldernames");
imdsTrain.ReadFcn = @(filename) im2single(imread(filename)); 
% inputSize=size(imdsTrain.readimage(1));


inputSize=[128 128 1];

augimds = augmentedImageDatastore([128 128], imdsTrain);
%%
%--------------------------------------------------------------------------
numEpochs = 80;
miniBatchSize = 32;
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
x=mbq.next;
mbq.reset;

monitor = trainingProgressMonitor;
monitor.Info = ["Epoch","LearnRate"];
monitor.Metrics = ["Loss","Mean_Grad_1","Mean_Grad_2","Mean_Grad_3","Mean_Grad_4"];
monitor.XLabel="Iteration";

groupSubPlot(monitor,"Loss","Loss");
groupSubPlot(monitor,"MeanGrad",["Mean_Grad_1","Mean_Grad_2","Mean_Grad_3","Mean_Grad_4"]);
app = ShowLearningIMGapp;
%%

% Find the indices of the weight learnables.
weightIdxE = ismember(netE.Learnables.Parameter,"Weights");
% Find the names of the layers with weights.
weightLayerNames = join([netE.Learnables.Layer(weightIdxE),...
    netE.Learnables.Parameter(weightIdxE)]);
plotSetup = setupGradientDistributionAxes(weightLayerNames,numEpochs);

%%
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
        % grad_trace=[grad_trace;gradNorm(gradientsE)];r

        % Update learnable parameters.
        [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
            gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);

        gradnorm=gradNorm(gradientsE);

        recordMetrics(monitor,iteration,Loss=loss,Mean_Grad_1=gradnorm(1),...
            Mean_Grad_2=gradnorm(2),Mean_Grad_3=gradnorm(3),Mean_Grad_4=gradnorm(4));
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs,LearnRate=learnRate);
        monitor.Progress = 100*iteration/numIterations;
    end


    z=predict(netE,x); y=predict(netD,z);
    new_img=y(:,:,:,1).extractdata;
    new_img=repmat(new_img, [1 1 3]);
    addImage(app, new_img);


 % At the end of each epoch, plot the gradient distributions of the weights
 % of each learnable layer using the supporting function
 % plotGradientDistributions.

 gradientValues = gradientsE.Value(weightIdxE);
 plotGradientDistributions(plotSetup,gradientValues,epoch)
end

save("Autoencoder.mat","netE","netD");