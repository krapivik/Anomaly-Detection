function A = sparse_act(X,p)
H=size(X,1); W=size(X,2);
for i=1:size(X,3)
    for j=1:size(X,4)   
        n=numel(find(X(:,:,i,j).extractdata>p));
        A(i,j)=n/(H*W);
        clear('n');
    end
end