function f = weightHist(net)
idx=find(net.Learnables.Parameter=="Weights");
n=numel(idx);
f=figure;
t=tiledlayout(1,n);
for i=1:n
    nexttile
    histogram(net.Learnables.Value{idx(i)}.extractdata)
end
title(t,"Weight histogram")
