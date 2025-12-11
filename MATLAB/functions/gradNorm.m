function gradNorm = gradNorm(grad)
idx=find(grad.Parameter=="Weights");
n=numel(idx);
for i=1:n
    gradNorm(i)=norm(grad.Value{idx(i)}.extractdata,"fro");
end
