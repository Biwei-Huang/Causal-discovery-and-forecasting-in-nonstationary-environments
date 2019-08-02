clear all,clc,close all
rng(10)
% consider the change of both causal coefficients and noise variance
% Xt = Bt*Xt + Et,
% b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});
% h_{i,t} = beta_{i}*h_{i,t-1} + eta_{i,t}, eta_t~N(0,p_{i});
addpath(genpath(pwd))

%% data generation
load generate_Data2_new % X_save B_save R_save A_save q_save G_save

T = 1000;
trial= 2;

X = X_save{trial}(:,1:T);
B0 = B_save{trial};
h0 = h_save{trial};
A0 = A_save{trial};
q0 = q_save{trial};
beta0 = beta_save{trial};
p0 = p_save{trial};
G0 = G_save{trial};

m = size(G0,1);         % number of variables
N1 = 20;                % Number of particles used in CPF-SAEM
numIter = 3500;          % Number of iterations in EM algorithms
kappa = 1;              % Constant used to compute SA step length (see below)

% SA step length
gamma = zeros(1,numIter);
gamma(1:2) = 1;
gamma(3:3399) = 0.96;
gamma(3400:end) = 0.96*(((0:numIter-3400)+kappa)/kappa).^(-0.4);

% if you have prior knowledge of the causal graph, you may modify B_Mask
% according to your prior knowledge. B_Mask(i,j) = 0 means the edge from i
% to j is fixed to zero.
B_Mask = ones(m,m)-eye(m);

% Initialization of the parameters
A_init = zeros(m,m);
q_init = zeros(m,m);
beta_init = 0.95*ones(m,1);
p_init = 0.05*ones(m,1);

% initialize the parameters by Kalman filter
[A_init,q_init] = KF_initialization(X);

% Run the algorithms
fprintf('Running CPF-SAEM (N=%i). Progress: ',N1); tic;
[q, A, beta, p, B,h] = cpf_saem2_new(numIter, X, N1, gamma,q_init,A_init,beta_init,p_init,B_Mask);
timeelapsed = toc;
fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);
% estimated parameters (use the parameters derived in the last iteration)
q_hat = q(:,:,end);
A_hat = A(:,:,end);
beta_hat = beta(:,end);
p_hat = p(:,end);
B_hat = B;
h_hat = h;
G_hat = ones(m,m); % the estimated causal graph
for i = 1:m
    for j = 1:m
        mu = mean(B_hat(i,j,:));
        va = var(B_hat(i,j,:));
        if(abs(mu)<0.05 & abs(va)<0.05)
            G_hat(i,j) = 0;
        end
    end
end
G_hat = G_hat';
save('example2', 'X','B0','h0','q0','A0','beta0','p0','G_hat','B_hat','h_hat','q_hat', 'A_hat', 'beta_hat','p_hat');

% plot each entry of B
figure,
for i = 1:m
    for j = 1:m
        subplot(m,m,(i-1)*m+j),plot(squeeze(B(j,i,1:end)));
    end
end

% plot each entry of h
figure,
for i = 1:m
    subplot(m,1,i),plot(h(i,1:end));
end


%%
% Forecasting by using the learned time-varying causal model
ts = T+1;
X_train = X_save{trial}(:,1:ts-1);
X_test = X_save{trial}(:,ts);
Data_sub = [X_train,X_test];

for j = 1:m
    target_id = j; % suppose we want to predict the next value of the j-th variable
    % prediction, y_pred saves the predicted values
    y_pred(target_id) = prediction_SSM2_new(G_hat,Data_sub',target_id,squeeze(B_hat(:,:,ts-1)),h_hat(:,ts-1),A_hat,q_hat,beta_hat,p_hat);
end
