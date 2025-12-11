function [loss, gradientsE, gradientsD] = modelLoss(netE,netD,X)

% Forward through encode
Z = forward(netE,X);
% Forward through decoder
Y = forward(netD,Z);
% Calculate loss and gradients.
% loss=1-mean(ssim(Y,X));
% loss=mse(Y,X);
loss=0.1*mse(Y,X)+20*(1-mean(ssim(Y,X)));
[gradientsE,gradientsD]=dlgradient(loss,netE.Learnables,netD.Learnables);

end