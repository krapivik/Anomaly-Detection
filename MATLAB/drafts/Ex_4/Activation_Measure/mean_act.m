function A = mean_act(X)
for i=1:size(X,3)
    for j=1:size(X,4)
        A(i,j)=mean(X(:,:,i,j),'all');
    end
end
