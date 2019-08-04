function v = matrix2vec(V,Mask)
% matrix to vector, remove the diagonal entries in the matrix
m = size(V,1);
r = sum(sum(Mask));
v = zeros(r,1);
count = 0;
for jj = 1:m % column
    for ii = 1:m % row
        if(Mask(ii,jj))
            count = count+1;
            v(count) = V(ii,jj);
        end
    end
end