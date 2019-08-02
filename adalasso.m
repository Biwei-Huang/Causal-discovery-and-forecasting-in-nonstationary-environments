function Mask = adalasso(X)
% adaptation lasso.

[N,T] = size(X);

% estimate the mask
Mask = zeros(N,N);
for i=1:N
    if T<4*N % sample size too small, so preselect the features
        tmp1 = X([1:i-1 i+1:N],:);
        [tmp2, Ind_t] = sort(abs(corr( tmp1', X(i,:)' )), 'descend');
        X_sel = tmp1(Ind_t(1:floor(N/4)),:); % pre-select N/4 features
        [beta_alt, beta_new_nt, beta2_alt, beta2_new_nt] = betaAlasso_grad_2step(X_sel, X(i,:), 0.65^2*var(X(i,:)), 20*log(T)/2); % 0.7^2
        beta2_al = zeros(N-1,1);
        beta2_al(Ind_t(1:floor(N/4))) = beta2_alt;
    else
        [beta_al, beta_new_n, beta2_al, beta2_new_n] = betaAlasso_grad_2step(X([1:i-1 i+1:N],:), X(i,:), 0.65^2*var(X(i,:)), 20*log(T)/2); % 0.7^2
    end
    Mask(i,1:i-1) = abs(beta2_al(1:i-1)) >0.005;
    Mask(i,i+1:N) = abs(beta2_al(i:N-1)) >0.005;
end
% Mask = Mask + Mask';
Mask = (Mask~=0);

