function [x]=SOMP(D,y,K); 
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).
%       y - the signals to represent
%       K - the max. number of coefficients for each signal.
% output arguments: 
%       x - sparse coefficient matrix.
% author: Cheng Wang, Hunan University
%=============================================
[m,l]=size(y);
[m,n]=size(D);
x = zeros(n,l);
%this algorithm reduces to standard orthogonal persuit when l=1;
residual = y;
index = [];
t = 1;

for t = 1:1:K,
    proj = D'*residual;
    
%     proj_t = sum(abs(proj),2);%1范数
   
 proj_t = max(proj',[],1);%每列的最大值
  
 if max(proj_t)> 0.0001    
    pos=find(proj_t==max(proj_t));
    pos=pos(1);
    index(t)=pos;
    x_temp=pinv(D(:,index))*y;
    residual=y-D(:,index)*x_temp;
 end
end
if (length(index)>0)
    x(index,:)=x_temp;
end
return;
