function ll = llf2(h,mu,q)
% calculate log-likelihood
m = size(h,1);
q = diag(q);
% ll = -1/2*(m*log(2*pi) + log(det(q)) + (h-mu)' * inv(q) * (h-mu));
% since q is diagnol
ll = -1/2*(m*log(2*pi) + sum(log(diag(q))) + (h-mu)' * inv(q) * (h-mu));
