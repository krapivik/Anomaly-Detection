function gradNorm = gradNorm(grad)
n=size(grad,1);
for i=1:n
    gradNorm(i)=norm(grad.Value{i}.extractdata,"fro");
end