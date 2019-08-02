function ll = llf1(B,mu,q,Mask)
% calculate log-likelihood
m = size(B,1);
r = sum(sum(Mask));
Bv = zeros(r,1); muv = zeros(r,1); qv = zeros(r,1);
count = 0;
for j = 1:m % column
    for i = 1:m % row
        if(Mask(i,j))
            count = count+1;
            Bv(count) = B(i,j);
            muv(count) = mu(i,j);
            qv(count) = q(i,j);
        end
    end
end
qv = diag(qv);
% ll = -1/2*(m*(m-1)*log(2*pi) + log(det(qv)) + (Bv-muv)' * inv(qv) * (Bv-muv));
% since qv is diagnol
ll = -1/2*(r*log(2*pi) + sum(log(diag(qv))) + (Bv-muv)' * inv(qv+1e-5*eye(r)) * (Bv-muv));