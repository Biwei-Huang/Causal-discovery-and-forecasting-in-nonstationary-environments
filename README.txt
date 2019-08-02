Copyright (c) 
 2018-2019 Biwei Huang

This package contains code to the paper for causal discovery and forecasting in nonstationary environments:
"Huang, B., Zhang, K., Gong, M., Glymour, C. Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models. ICML 2019."

The code is written in Matlab R2017a.


%%%%%%%%%%%%%
IMPORTANT FUNCTIONS
%%%%%%%%%%%%%
function [R, q, A, B] = cpf_saem1_new(numIter, X, N, gamma,R_init,q_init,A_init,B_Mask)
% Time-varying causal model estimation when only considering the change of causal coefficients
% Xt = Bt*Xt + Et, Et~N(0,R);
% b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});

% Input:
%  numIter: number of Iterations
%  X: observed data
%  N: number of sampled particles in each iteration
%  gamma: step length of stochastic approximation
%  R_init: initiation value of the noise variance
%  q_init: initialion value of the noise variance in the auto-regressive model of B
%  A_init: initiation value of the coefficients in the auto-regressive model of B
%  B_Mask: a pre-defined Mask of causal connectivity

% Output:
%  R: estimated noise variance
%  q: estimated noise variance in the auto-regressive model of B
%  A: estimated coefficients in the auto-regressive model of B
%  B: estimated time-varying causal strength


function [q, A, beta, p, B,h] = cpf_saem2_new(numIter, X, N, gamma,q_init,A_init,beta_init,p_init,B_Mask)
% Time-varying causal model estimation when considering the change of causal coefficients and noise variance
% Xt = Bt*Xt + Et,
% b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});
% h_{i,t} = beta_{i}*h_{i,t-1} + eta_{i,t}, eta_t~N(0,p_{i});

% Input:
%  numIter: number of Iterations
%  X: observed data
%  N: number of sampled particles in each iteration
%  gamma: step length of stochastic approximation
%  q_init: initialion value of the noise variance in the auto-regressive model of B
%  A_init: initiation value of the coefficients in the auto-regressive model of B
%  beta_init: initialion value of the noise variance in the auto-regressive model of h
%  p_init: initiation value of the coefficients in the auto-regressive model of h
%  B_Mask: a pre-defined Mask of causal connectivity

% Output:
%  q: estimated noise variance in the auto-regressive model of B
%  A: estimated coefficients in the auto-regressive model of B
%  beta: estimated noise variance in the auto-regressive model of h
%  p: estimated coefficients in the auto-regressive model of h
%  B: estimated time-varying causal strength
%  h: estimated time-varying log-variance of the noise


function y_star = prediction_SSM1_new(G,Data,target_id,Bt,A,q,R)
% Forecasting (only considering the change of causal coefficients)

% Input:
%  G: causal graph
%  D: contain all data from 1 to time T, and from time T+1
%  target_id: the index of the variable that we want to predict
%  Bt: causal strength at time T
%  A: estimated coefficients in the auto-regressive model of B
%  q: estimated noise variance in the auto-regressive model of B
%  R: estimated noise variance

% Output:
%  y_star: the predicted value at time T+1 of variable with target_id


function y_star = prediction_SSM2_new(G,Data,target_id,Bt,ht,A,q,beta,p)
% Forecasting (considering both the change of causal coefficients and noise variance)

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



%%%%%%%%%%%%%
EXAMPLE
%%%%%%%%%%%%%
example1.m and example2.m give two example of using this package.



%%%%%%%%%%%%%
CITATION
%%%%%%%%%%%%%
If you use this code, please cite the following paper:
"Huang, B., Zhang, K., Gong, M., Glymour, C. Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models. ICML 2019."



If you have problems or questions, do not hesitate to send an email to biweih@andrew.cmu.edu
