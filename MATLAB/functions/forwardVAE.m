function [Y,mu,logSigmaSq]=forwardVAE(net,X)

mu=forward(net,X, "Outputs","mean");
logSigmaSq=forward(net,X,"Outputs","log-variance");
Y = forward(net, X);
end