function X = preprocessMiniBatch(dataX)

% Concatenate.
X = cat(4,dataX{:});
% X = im2double(X);         % нормируем [0,1]
% X = dlarray(X, "SSCB");   % формируем dlarray для сети
end