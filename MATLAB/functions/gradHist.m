function f = gradHist(grad)
idx=find(grad.Parameter=="Weights");
n=numel(idx);
f=figure;
t=tiledlayout(1,n);
for i=1:n
    if grad.Parameter{idx(i)}=="Weights"
        max_grad=max(abs(grad.Value{i}.extractdata),[],"all");
        mean_grad=mean(grad.Value{i}.extractdata,"all");
        std_grad=std(grad.Value{i}.extractdata,[],"all");
    nexttile
    histogram(grad.Value{i}.extractdata)
    title([grad.Layer{idx(i)} grad.Parameter{idx(i)}])
    subtitle([strcat("Abs Max = ",num2str(max_grad)) ...
        strcat("Mean = ",num2str(mean_grad)) ...
        strcat("STD = ",num2str(std_grad))])
    end
end
title(t,"Grad Histogram")