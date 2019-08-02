% Generate data from this process, and try to learn the dynamics back.

% B(t+1) = F B(t) + noise(Q), A diagonal
% Y(t) = B(t) X(t) + noise(R)
clear all,clc,close all
addpath('../KPMstats/')
addpath('../KPMtools/')

T = 1000;
ss = 2; % state size
os = 1; % observation size
F = 0.8*eye(ss);
X = randn(os, ss, T);
Q = 0.1*eye(ss);
R = 1*eye(os);

F = repmat(F, 1, 1, T);
Q = repmat(Q, 1, 1, T);
R = repmat(R, 1, 1, T);

initx = zeros(ss,1);
initV = 1*eye(ss);

seed = 1;
rand('state', seed);
randn('state', seed);
[B,y] = sample_lds(F, X, Q, R, initx, initV, T, 1:T);

% Initializing the params to sensible values is crucial.
% Here, we use the true values for everything except F and H,
% which we initialize randomly (bad idea!)
% Lack of identifiability means the learned params. are often far from the true ones.
% All that EM guarantees is that the likelihood will increase.
F1 = eye(ss);
F1 = repmat(F1, 1, 1, T);
X1 = X;
Q1 = Q;
R1 = R;
initx1 = initx;
initV1 = initV;
max_iter = 100;
[F2, X2, Q2, R2, initx2, initV2, LL] =  learn_kalman_nonstationary(y, F1, X1, Q1, R1, initx1, initV1, 1:T, max_iter, 1, 1);

% [Best, V, VV, loglik] = kalman_filter(y, F, X, Q, R, initx, initV, 'model', 1:T);
[Best, V, VV, loglik] = kalman_filter(y, F2, X2, Q2, R2, initx2, initV2, 'model', 1:T);
Bpred = squeeze(F(:,:,end))*Best(:,end); % the prediction point