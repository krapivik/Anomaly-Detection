function f = errorMap(X,Y)
err=abs(X-Y);
f=figure;
tiledlayout(1,3)
nexttile
imshow(X)
nexttile
imshow(Y)
nexttile
imshow(err);