function i = resampling(q)
N = length(q);
% Multinomial resampling
u = rand(N,1);
qc = cumsum(q);
qc = qc(:);
qc=qc/qc(N);
[~,ind1]=sort([u;qc]);
ind2=find(ind1<=N);
i=ind2'-(0:N-1);
end