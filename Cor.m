function  [Corrcoef] = Cor(train_data_ori, test_data_ori, train_label,F)

[m, n] = size(test_data_ori);
Corrcoef = zeros(max(train_label), n);
for p = 1:n
    y = test_data_ori(:, p);
    for i = 1:max(train_label)
        A_1 = [];
        X1 = train_data_ori(:, find(train_label == i));
        K = F;
        if size(X1,2)<K
            K = size(X1,2);
        else
            K = F;
        end
        for z = 1:size(X1,2)
            A = corr2(y,X1(:,z));
            A_1(z) = A;
        end
        sort_d = sort(A_1, 'descend');                  
        Corrcoef(i, p) = 1- sum(sort_d(1:K)) / K;        
    end
end