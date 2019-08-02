function [R, q, A, B] = cpf_saem1_new(numIter, X, N, gamma,R_init,q_init,A_init,B_Mask)
% only consider the change of causal coefficients
% Xt = Bt*Xt + Et, Et~N(0,R);
% b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});

% Runs the CPF-SAEM algorithm (based on the following paper: Lindsten, F. An efficient stochastic approximation EM algorithm using conditional particle filters.ICASSP,2013.)
% Derive the update of R,q,A at M step directly
% The particles in each iteration are not saved (only the particles in the last iteration are saved)

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
 
T = size(X,2); % the length of time series
m = size(X,1); % the number of observed time series
r = sum(sum(B_Mask));

R = zeros(m,m,numIter);
q = zeros(m,m,numIter);
A = zeros(m,m,numIter);

B = zeros(m,m,T);

% Initialize the parameters
R(:,:,1) = R_init;
q(:,:,1) = q_init;
A(:,:,1) = A_init;

% Initialize the state by running a PF
[particles, w] = cpf_as(X,N,squeeze(R(:,:,1)),squeeze(q(:,:,1)),squeeze(A(:,:,1)),B,B_Mask);
% Draw J
J = find(rand(1) < cumsum(w(:,T)),1,'first');
B = squeeze(particles(J,:,:,:));

% parameters
S = [];
S.R = zeros(m,m);
S.Ra = 0;
S.Rb = zeros(m,m);
S.A = zeros(r,r);
S.Aa = zeros(r,r);
S.Ab = zeros(r,r);
S.q = zeros(r,r);
S.qa = 0;
S.qb = zeros(r,r);

% Run identification loop
for k = 2:numIter
    % Update (M step)
    S = update_M(S,particles, w(:,T), gamma(k), X,B_Mask);
    R(:,:,k) = S.R;
    q(:,:,k) = vec2matrix(diag(S.q),B_Mask);
    A(:,:,k) = vec2matrix(diag(S.A),B_Mask);
    
    % check whether it has been converged
    diffq = sum(sum(abs(squeeze(q(:,:,k))-squeeze(q(:,:,k-1)))));
    diffA = sum(sum(abs(squeeze(A(:,:,k))-squeeze(A(:,:,k-1)))));
    diffR = sum(sum(abs(squeeze(R(:,:,k))-squeeze(R(:,:,k-1)))));
    fprintf('Iter = %d, %.4f\n',k,diffq + diffA + diffR);
    diffqAR(k) = (diffq + diffA + diffR);
%     if(diffqAR<0.01)
%         break;
%     end
    
    
    % Run CPF-AS
    [particles, w] = cpf_as(X,N,squeeze(R(:,:,k)),squeeze(q(:,:,k)),squeeze(A(:,:,k)),B,B_Mask);
    
    % Draw J (extract a particle trajectory)
    J = find(rand(1) < cumsum(w(:,T)),1,'first');
    B = squeeze(particles(J,:,:,:));
    
end
end
%--------------------------------------------------------------------------
function [z,w] = cpf_as(X,N,R,q,A,B,B_Mask)
% Conditional particle filter with ancestor sampling
% Input:
%   X - measurements
%   N - number of particles
%   R - measurement noise variance
%   q - process noise variance
%   B - conditioned particles - if not provided, un unconditional PF is run

conditioning = 1;
T = size(X,2);
m = size(X,1);
z = zeros(N,m,m,T); % Particles
a = zeros(N, T); % Ancestor indices
w = zeros(N, T); % Weights
z(:,:,:,1) = 0; % Deterministic initial condition
z(N,:,:,1) = B(:,:,1); % Set the 1st particle according to the conditioning

for t = 1:T
    if(t==1)
        for i = 1:N
            z(i,:,:,t) = sqrt(q).*randn(m,m);
        end
        if(conditioning)
            z(N,:,:,t) = B(:,:,t); % Set the N-th particle according to the conditioning
        end
        % Compute importance weights
        LogLikelihoods = [];
        for i = 1:N
            %             E(:,i,t) = X(:,t)-squeeze(z(i,:,:,t))*X(:,t); % E: m*N
            tmp = inv(eye(m)-squeeze(z(i,:,:,t)));
            Cov_X = tmp * R * tmp';
            LogLikelihoods(i) = -0.5*(m*log(2*pi) + log(det(Cov_X)) + X(:,t)' * inv(Cov_X) * X(:,t));
        end
        %         weights = exp(LogLikelihoods);
        %         w(:,t) = weights'/sum(weights); % Save the normalized weights
        for i = 1:N
            w(i,t) = 1/(sum(exp(LogLikelihoods-LogLikelihoods(i))));
        end
    else
        ind = resampling(w(:,t-1));
        ind = ind(randperm(N));
        for i = 1:N
            zpred(i,:,:) = A.*squeeze(z(i,:,:,t-1));
        end
        for i = 1:N
            z(i,:,:,t) = squeeze(zpred(ind(i),:,:)) + sqrt(q).*randn(m,m);
        end
        if(conditioning)
            z(N,:,:,t) = B(:,:,t); % Set the N:th particle according to the conditioning
            % Ancestor sampling
            for i = 1:N
                ll(i) = llf1(squeeze(B(:,:,t)),squeeze(zpred(i,:,:)),q,B_Mask);
            end
            w_as = w(:,t-1).*exp(ll)';
            w_as = w_as/sum(w_as);
            ind(N) = find(rand(1) < cumsum(w_as),1,'first');
        end
        % Store the ancestor indices
        a(:,t) = ind;
        
        % Compute importance weights
        LogLikelihoods = [];
        for i = 1:N
            %             E(:,i,t) = X(:,t)-squeeze(z(i,:,:,t))*X(:,t); % E: m*N
            tmp = inv(eye(m)-squeeze(z(i,:,:,t)));
            Cov_X = tmp * R * tmp';
            LogLikelihoods(i) = -0.5*(m*log(2*pi) + log(det(Cov_X)) + X(:,t)' * inv(Cov_X) * X(:,t));
        end
        %         weights = exp(LogLikelihoods);
        %         w(:,t) = weights/sum(weights); % Save the normalized weights
        for i = 1:N
            w(i,t) = 1/(sum(exp(LogLikelihoods-LogLikelihoods(i))));
        end
    end
end

% Generate the trajectories from ancestor indices
ind = a(:,T);
for t = T-1:-1:1
    z(:,:,:,t) = z(ind,:,:,t);
    ind = a(ind,t);
end
end


function S = update_M(S,particles, w, gamma, X,B_Mask)
N = length(w); % number of particles
B = particles;
T = size(X,2);
m = size(X,1);
r = sum(sum(B_Mask));

Ra_tmp = 0;
Rb_tmp = zeros(m,m);

Aa_tmp = zeros(r,r);
Ab_tmp = zeros(r,r);
qa_tmp = 0;
qb_tmp = zeros(r,r);
for i = 1:N
    for t = 1:T
        Bt = squeeze(B(i,:,:,t));
        Ra_tmp = Ra_tmp + w(i);
        Rb_tmp = Rb_tmp + w(i)*(eye(m)-Bt)*X(:,t)*X(:,t)'*(eye(m)-Bt)';
        
        if(t>1)
            Bt1 = squeeze(B(i,:,:,t-1));
            Bvt1 = matrix2vec(Bt1,B_Mask);
            Bvt = matrix2vec(Bt,B_Mask);
            qa_tmp = qa_tmp + w(i);
            qb_tmp = qb_tmp + w(i)*(Bvt-S.A*Bvt1)*(Bvt-S.A*Bvt1)';
            Aa_tmp = Aa_tmp + w(i)*Bvt*Bvt1';
            Ab_tmp = Ab_tmp + w(i)*Bvt1*Bvt1';
        end
    end
end
S.Ra = (1-gamma)*S.Ra + gamma*Ra_tmp;
S.Rb = diag(diag((1-gamma)*S.Rb + gamma*Rb_tmp));
S.R = S.Rb/S.Ra;

S.qa = (1-gamma)*S.qa + gamma*qa_tmp;
S.qb = diag(diag((1-gamma)*S.qb + gamma*qb_tmp));
S.q = S.qb/S.qa;

S.Aa = (1-gamma)*S.Aa + gamma*Aa_tmp;
S.Ab = (1-gamma)*S.Ab + gamma*Ab_tmp;
S.A = diag(diag(S.Aa*inv(S.Ab)));


end

