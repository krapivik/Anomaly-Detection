classdef myReshapeLayer < nnet.layer.Layer  & nnet.layer.Formattable % (Optional) 
    properties
        TargetSize  % Размерность выходного тензора [H W C] 
    end
    
    % properties (Learnable)
    %     Weights
    %     Bias
    % end

    methods
        function layer = myReshapeLayer(targetSize,name) 
            layer.TargetSize=targetSize;
            layer.Name = name;
            layer.Description = "Reshape vector to matrix of size " + mat2str(targetSize);
        end

        function Z = predict(layer,X)

            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data, specified as a formatted dlarray
            %                 with a "C" and optionally a "B" dimension.
            % Outputs:
            %         Z     - Output of layer forward function returned as
            %                 a formatted dlarray with format "SSCB".
            
            % weights = layer.Weights;
            % bias = layer.Bias;
            % X = fullyconnect(X,weights,bias);

            targetSize=layer.TargetSize;
            batchSize=size(X,2);
            x=extractdata(X);
            Z=[];
            for i=1:batchSize
            Z(:,:,:,i)=reshape(x(:,i), targetSize); 
            end     
            Z = dlarray(Z,"SSCB");
        end
    end
end