function [A_all, Q_all,R_all] = KF_initialization(Data)
%   y(t)   = C*x(t) + v(t),  v ~ N(0, R)
%   x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
% [A, C, Q, R, INITX, INITV, LL] = LEARN_KALMAN(DATA, A0, C0, Q0, R0, INITX0, INITV0) fits

addpath('./KPMstats/')
addpath('./KPMtools/')
addpath('./Kalman/')

[m,T] = size(Data);
A_all = [];
Q_all = [];
R_all = [];

for j = 1:m
    y_id = j;
    y = Data(y_id,:);
    X = Data(setdiff(1:m,y_id),:);
    
    A_init = 0.5*eye(m-1);
    A_init = repmat(A_init, 1, 1, T);
    X_tr = zeros(1,m-1,T);
    X_tr(1,:,:) = X;
    R_init = 0.5*eye(1);
    Q_init = 0.5*eye(m-1);
    initx1 = zeros(m-1,1);
    initV1 = 0.2*eye(m-1);
    max_iter = 100;
    Q_init = repmat(Q_init, 1, 1, T);
    R_init = repmat(R_init, 1, 1, T);
    
    [A, C, Q, R, initx2, initV2, LL] =  learn_kalman_nonstationary(y, A_init, X_tr, Q_init, R_init, initx1, initV1, 1:T, max_iter, 1, 1);
    
    A = diag(squeeze(A(:,:,1)));
    Q = diag(squeeze(Q(:,:,1)));
    R = squeeze(R(:,:,1));
    A_all = [A_all; A];
    Q_all = [Q_all; Q];
    R_all = [R_all;R];
end

Mask = ones(m,m)-eye(m);
A_all = vec2matrix(A_all,Mask)';
Q_all = vec2matrix(Q_all,Mask)';
R_all = diag(R_all);


