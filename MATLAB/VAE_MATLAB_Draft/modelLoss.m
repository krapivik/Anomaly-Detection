function [loss, gradients] = modelLoss(net,X)

[Y,mu,logSigmaSq] = forwardVAE(net,X);


% Calculate loss and gradients.
loss = elboLoss(Y,X,mu,logSigmaSq);
gradients = dlgradient(loss,net.Learnables);
end

function loss = elboLoss(Y,T,mu,logSigmaSq)

reconstructionLoss=mse(Y,T);
% KL divergence.
KL = -0.5 * sum(1 + logSigmaSq - mu.^2 - exp(logSigmaSq),1);
KL = mean(KL);

disp(extractdata(KL));
disp(extractdata(reconstructionLoss));

% Combined loss.
loss = reconstructionLoss + KL;
end