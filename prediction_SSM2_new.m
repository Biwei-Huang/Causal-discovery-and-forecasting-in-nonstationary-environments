% Forecasting: predict Y(t+1)
% consider the change of both causal coefficients B and logarithm of noise variance h
% using Metropolics-Hasting and Monte Carlo integration
% this code is much more efficient
function y_star = prediction_SSM2_new(G,Data,target_id,Bt,ht, A,q,beta,p)
% Input:
%  G: causal graph
%  D: contain all data from 1 to time T, and from time T+1
%  target_id: the index of the variable that we want to predict
%  Bt: estimated causal strength at time T
%  ht: estimated log-variance of the noise at time T
%  A: estimated coefficients in the auto-regressive model of B
%  q: estimated noise variance in the auto-regressive model of B
%  beta: estimated noise variance in the auto-regressive model of h
%  p: estimated coefficients in the auto-regressive model of h

% Output:
%  y_star: the predicted value at time T+1 of variable with target_id

yt = Data(end-1,target_id); % value at time t

pa_id = find(G(:,target_id)==1); % parents
ch_id = find(G(target_id,:)==1)'; % children

% sampling B(t+1), h(t+1)
N = 10000; % number of samples to sample
% y|pa
if(~isempty(pa_id))
    b_y = mvnrnd(A(target_id,pa_id).*Bt(target_id,pa_id),diag(q(target_id,pa_id)),N);
else
    b_y = 0;
end
h_y = beta(target_id)*ht(target_id) + sqrt(p(target_id))*randn(N,1);
% ch|y,sp
pch_id = [];
b_ch = [];
h_ch = [];
for j = 1:length(ch_id)
    pch_id{j} = find(G(:,ch_id(j))==1);% parents of children, except the target
    pch_id{j} = setdiff(pch_id{j},target_id);
    
    b_ch{j} = mvnrnd(A(ch_id(j),[target_id;pch_id{j}]).*Bt(ch_id(j),[target_id;pch_id{j}]),diag(q(ch_id(j),[target_id;pch_id{j}])),N);
    h_ch{j} = beta(ch_id(j))*ht(ch_id(j)) + sqrt(p(ch_id(j)))*randn(N,1);
end

% Metropolics-Hasting
sigma = [];  % standard deviation of proposed distribution
while(1)
    if(isempty(sigma))
        sigma = std(Data(:,target_id))*3;  % standard deviation of proposed distribution
    end
    y = [];
    y(1) = yt;% initialize y
    for n = 2:10000
        % propose
        y_candi = y(n-1) + sigma*randn(1);
        
        % acceptance probability a
        % estimate p(y|pa) and p(ch|y,sp) by Monte Carlo
        if(~isempty(pa_id))
            pr_y_candi = normpdf(y_candi,(b_y*Data(end,pa_id)')',sqrt(exp(h_y)'));
            pr_y1 = normpdf(y(n-1),(b_y*Data(end,pa_id)')',sqrt(exp(h_y)'));
        else
            pr_y_candi = normpdf(y_candi,0,sqrt(exp(h_y)'));
            pr_y1 = normpdf(y(n-1),0,sqrt(exp(h_y)'));
        end
        for j = 1:length(b_ch)
            pr_ch_candi(j,:) = normpdf(Data(end,ch_id(j)),(b_ch{j}*[y_candi;Data(end,pch_id{j})'])',sqrt(exp(h_ch{j})'));
            pr_ch1(j,:) = normpdf(Data(end,ch_id(j)),(b_ch{j}*[y(n-1);Data(end,pch_id{j})'])',sqrt(exp(h_ch{j})'));
        end
        
        a = mean(pr_y_candi)/mean(pr_y1);
        for j = 1:length(ch_id)
            a = a*mean(pr_ch_candi(j,:))/mean(pr_ch1(j,:));
        end
        a = min(1,a);
        u = rand;
        if(u<a)
            y(n) = y_candi; % accept the proposal
            sign(n) = 1;
        else
            y(n) = y(n-1); % reject the proposal
            sign(n) = 0;
        end
    end
    
    s = sum(sign)/length(sign)
    if(s<=0.3)
        break;
    else
        sigma = sigma*2;
    end
end

sum(sign)/length(sign)
y_star = mean(y(1001:end)); % ignore the first 1000 samples
% var_y_star = var(y(1001:end))*length(y(1001:end));
% var_y_star = var(y(1001:end));
% y_star = [mean_y_star;var_y_star];







