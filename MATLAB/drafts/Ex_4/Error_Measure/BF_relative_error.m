function [back_err, front_err]=BF_relative_error(X,Y)

eps=0.01;
idx_back=find(X.extractdata<=eps); % Фон исходного изображения
idx_front=find(X.extractdata>eps); % Пиксели числа исходного изображения
idx_back_rec=find(Y.extractdata<=eps); % Фон реконструированного изображения
idx_front_rec=find(Y.extractdata>eps); % Пиксели реконструированного числа

% back_intersection=intersect(idx_back,idx_back_rec);
back_union_idx=union(idx_back,idx_back_rec);

% front_intersection=intersect(idx_front,idx_front_rec);
front_union_idx=union(idx_front,idx_front_rec);

back_err_pix=numel(setdiff(back_union_idx,idx_back));
front_err_pix=numel(setdiff(front_union_idx,idx_front));

back_err=back_err_pix/numel(idx_back);
front_err=front_err_pix/numel(idx_front_rec);
