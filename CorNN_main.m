%% ==================================================
%              Written by xiaofei zhang             %
%%===================================================
close all;clear all;
clc;

C = 16;K = 1;F = 4;scale = 6;
lambda_cor = 0.6;
load IndiaP
im_2d = ToVector(img)';
fimage = img;
load Indian_pines_gt
im_gt = indian_pines_gt;
[i_row, i_col] = size(im_gt);
im_gt_1d = reshape(im_gt,1,i_row*i_col);
index_map = reshape(1:length(im_gt_1d),[i_row,i_col]);
%==================================================================================
index = [];label = [];
num_class = [];
for i = 0:1:C
    index_t =  find(im_gt_1d == i);
    index = [index index_t];
    label_t = ones(1,length(index_t))*i;
    label = [label label_t];
    num_class_t = length(index_t);
    num_class = [num_class num_class_t];
end
CA_new=[];
num_tr = [1048,12,140,83,20,50,73,5,50,5,100,250,60,20,130,40,10];  %10%

D = [];D_label = [];tt_data = [];tt_label = [];
tt_index = [];temp_train = [];temp_test = [];
for i = 1:1:C
    label_c = find(label == i);
    random_index = label_c(randperm(length(label_c)));
    temp = index(random_index(1:num_tr(i+1)));
    temp_train = [temp_train temp];
    D_i = im_2d(:,temp);
    D = [D D_i];
    D_label_i = ones(1,length(temp))*i;
    D_label = [D_label D_label_i];
    temp = index(random_index(num_tr(i+1)+1:end));
    tt_data_i = im_2d(:,temp);
    temp_test = [temp_test temp];
    tt_data = [tt_data tt_data_i];
    tt_label_i = ones(1,length(temp))*i;
    tt_label = [tt_label tt_label_i];
    tt_index = [tt_index temp];
end
data_all = [D,tt_data ];
labels = [D_label,tt_label];
D = D./repmat(sqrt(sum(D.*D)),[size(D,1) 1]);
label_result = zeros(size(tt_label));
train_data_ori = data_all(:, (1:num_tr(1)));
test_data_ori = data_all(:, (num_tr(1)+1:end));
%================================Residual of JSRC================================
for i = 1:1:size(tt_data,2)
    row = mod(tt_index(i),i_row);
    if row == 0
        row = i_row;
    end
    col = ceil(tt_index(i)/i_row);
    row_range = ceil(row-(scale-1)/2 : row+(scale-1)/2);
    row_range(row_range<=0)= 1;row_range(row_range>=i_row)= i_row;
    col_range = ceil(col-(scale-1)/2 : col+(scale-1)/2);
    col_range(col_range<=0)= 1;col_range(col_range>=i_col)= i_col;
    temp = fimage(row_range,col_range,:);
    X = ToVector(temp)';
    X = X./repmat(sqrt(sum(X.*X)),[size(X,1) 1]);
    S = SOMP(D,X,K);
    for j = 1:1:C
        temp = find(D_label == j);
        D_c = D(:,temp);
        S_c = S(temp,:);
        re_temp = X - D_c*S_c;
        residual(i,j) = norm(re_temp,'fro');
    end
end
residual_1 = residual./repmat(sqrt(sum(residual.*residual)),[size(residual,1) 1]);
%===============================Corrcoef=============================
Corrcoef_1 = Cor(train_data_ori, test_data_ori, D_label, F);
% =============================Classification=============================
w_cor = residual_1' + lambda_cor * Corrcoef_1;
for i = 1:length(tt_label)
    result(i) = find(w_cor(:, i) == min(w_cor(:, i)));
end
[OA,AA,kappa,CA] = confusion(tt_label, result);

im_gt_map = im_gt;
im_gt_map(tt_index) = result;
figure()
map = label2color(im_gt_map,'india');
imshow(map);
