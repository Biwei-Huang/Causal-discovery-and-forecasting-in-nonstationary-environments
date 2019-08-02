function V = vec2matrix(v,Mask)
% vector to matrix, the diagonal entries in the matrix are zero
m = size(Mask,1);
V = zeros(m,m);
count = 0;
for j = 1:m
    for i = 1:m
        if(Mask(i,j))
            count = count+1;
            V(i,j) = v(count);
        end
    end
end
            
            


