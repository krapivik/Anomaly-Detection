function f = featureMap(net, X, layer)
feature=minibatchpredict(net,X,'Outputs',layer);
n=size(feature,3);
all_pixels=size(feature,1)*size(feature,2);
h=sqrt(n);
f=figure;
eps=0.3;
if rem(h,1)==0
    t=tiledlayout(h,h);
    title(t, ["Активации" layer])
    for i=1:n
        active_pixels=numel(find(feature(:,:,i,1).extractdata>eps));
        relative_activation(i)=active_pixels/all_pixels;
        nexttile
        imshow(feature(:,:,i,1).extractdata)
        title(["Отн. активация: " relative_activation(i)])
    end
else
    h=sqrt(n/2);
    t=tiledlayout(h,h*2);
    title(t, ["Активации" layer])
    for i=1:n
        active_pixels=numel(find(feature(:,:,i,1).extractdata>eps));
        relative_activation(i)=active_pixels/all_pixels;
        nexttile
        imshow(feature(:,:,i,1).extractdata)
        title(["Отн. активация: " relative_activation(i)])
    end   
end
activated_maps=numel(find(relative_activation>eps));
subtitle(t, ["Количество активных карт признаков: " activated_maps])