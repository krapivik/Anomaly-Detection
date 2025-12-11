function [loss, gradientsE, gradientsD] = modelLoss(netE,netD,X)

% Forward through encode
Z = forward(netE,X);
% Forward through decoder
Y = forward(netD,Z);
% Calculate loss and gradients.
loss=mse(Y,X);
[gradientsE,gradientsD]=dlgradient(loss,netE.Learnables,netD.Learnables);
end

