function [centers,counts] = gradientDistributions(values)
% Get the histogram count for the values.
[counts,edges] = histcounts(values,30);

% histcounts returns edges of the bins. To get the bin centers,
% calculate the midpoints between consecutive elements of the edges.
centers =  edges(1:end-1) + diff(edges)/2;
end