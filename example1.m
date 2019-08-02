clear all,clc,close all
rng(100)
% only consider the change of causal coefficients
% Xt = Bt*Xt + Et, Et~N(0,R);
% b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});

addpath(genpath(pwd))

%% data generation
load generate_Data1_new % X_save B_save R_save A_save q_save G_save

T = 1000;
trial = 1;

X = X_save{trial}(:,1:T);
B0 = B_save{trial}; % true parameters
R0 = R_save{trial};
A0 = A_save{trial};
q0 = q_save{trial};
G0 = G_save{trial};

m = size(G0,1);         % number of variables
N1 = 20;                % Number of particles used in CPF-SAEM
numIter = 4000;          % Number of iterations in EM algorithms 
% one may reduce the number of iterations to speed up the process
kappa = 1;              % Constant used to compute SA step length (see below)

% SA step length
gamma = zeros(1,numIter);
gamma(1:2) = 1;
gamma(3:3499) = 0.96;
gamma(3500:end) = 0.96*(((0:numIter-3500)+kappa)/kappa).^(-0.4);

% if you have prior knowledge of the causal graph, you may modify B_Mask
% according to your prior knowledge. B_Mask(i,j) = 0 means the edge from i
% to j is fixed to zero.
B_Mask = ones(m,m);
B_Mask = B_Mask - eye(m);

% Initialization of the parameters
A_init = zeros(m,m);
q_init = zeros(m,m);
R_init = diag(0.2*ones(m,1));

% initialize the parameters by Kalman filter
[A_init,q_init,R_init] = KF_initialization(X);


% Run the algorithms
fprintf('Running CPF-SAEM (N=%i). Progress: ',N1); tic;
[R,q,A,B] = cpf_saem1_new(numIter, X, N1, gamma,R_init,q_init,A_init,B_Mask);
timeelapsed = toc;
fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);
% estimated parameters (use the parameters derived in the last iteration)
R_hat = R(:,:,end);
q_hat = q(:,:,end);
A_hat = A(:,:,end);
B_hat = B;
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
save('example1', 'X','B0','q0','A0','R0','G_hat','B_hat', 'q_hat', 'A_hat', 'R_hat');

% plot each entry of B
figure,
for j = 1:m
    for i = 1:m
        if(i~=j)
            subplot(m,m,(i-1)*m+j),plot(squeeze(B(j,i,1:end)));
        end
    end
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
    y_pred(target_id) = prediction_SSM1_new(G_hat,Data_sub',target_id,squeeze(B_hat(:,:,ts-1)),A_hat,q_hat,R_hat);
end


