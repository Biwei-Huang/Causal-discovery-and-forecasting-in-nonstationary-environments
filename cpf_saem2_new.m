function [q, A, beta, p, B,h] = cpf_saem2_new(numIter, X, N, gamma,q_init,A_init,beta_init,p_init,B_Mask)
% consider the change of both causal coefficients and noise variance
% Xt = Bt*Xt + Et,
% b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});
% h_{i,t} = beta_{i}*h_{i,t-1} + eta_{i,t}, eta_t~N(0,p_{i});

% Runs the CPF-SAEM algorithm (based on the following paper: Lindsten, F. An efficient stochastic approximation EM algorithm using conditional particle filters.ICASSP,2013.)
% Derive the update of q,A,beta,p at M step directly
% Use stochastic volatility model to model the noise variance
% The particles in each iteration are not saved (only the particles in the last iteration are saved)

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

T = size(X,2); % the length of time series
m = size(X,1); % the number of observed time series
r = sum(sum(B_Mask));

q = zeros(m,m,numIter);
A = zeros(m,m,numIter);
beta = zeros(m,numIter);
p = zeros(m,numIter);

B = zeros(m,m,T);
h = zeros(m,T);

% Initialize the parameters
q(:,:,1) = q_init;
A(:,:,1) = A_init;
beta(:,1) = beta_init;
p(:,1) = p_init;


% Initialize the state by running a PF
[particles1,particles2,w] = cpf_as(X,N,squeeze(q(:,:,1)),squeeze(A(:,:,1)),beta(:,1),p(:,1),B,h,B_Mask);
% Draw J
J1 = find(rand(1) < cumsum(w(:,T)),1,'first');
B = squeeze(particles1(J1,:,:,:));
h = squeeze(particles2(J1,:,:));

% parameters
S = [];
S.A = zeros(r,r);
S.Aa = zeros(r,r);
S.Ab = zeros(r,r);
S.q = zeros(r,r);
S.qa = 0;
S.qb = zeros(r,r);
S.beta = zeros(m,m);
S.betaa = zeros(m,m);
S.betab = zeros(m,m);
S.p = zeros(m,m);
S.pa = 0;
S.pb = zeros(m,m);

% Run identification loop
for k = 2:numIter
    % Update (M step)
    S = update_M(S,particles1,particles2, w(:,T), gamma(k), X,B_Mask);
    q(:,:,k) = vec2matrix(diag(S.q),B_Mask);
    A(:,:,k) = vec2matrix(diag(S.A),B_Mask);
    beta(:,k) = diag(S.beta);
    p(:,k) = diag(S.p);
    
    % check whether it is converged
    diffq = sum(sum(abs(squeeze(q(:,:,k))-squeeze(q(:,:,k-1)))));
    diffA = sum(sum(abs(squeeze(A(:,:,k))-squeeze(A(:,:,k-1)))));
    diffbeta = sum(abs(beta(:,k)-beta(:,k-1)));
    diffp = sum(abs(p(:,k)-p(:,k-1)));
    fprintf('Iter = %d, %.4f\n',k,diffq + diffA + diffbeta + diffp);
    diffqbp(k) = diffq + diffbeta + diffA + diffp;
    %     if(diffq + diffA + diffbeta + diffp < 1e-6)
    %         break;
    %     end

    % Run CPF-AS
    [particles1,particles2, w] = cpf_as(X,N,squeeze(q(:,:,k)),squeeze(A(:,:,k)),beta(:,k),p(:,k),B,h,B_Mask);
    
    % Draw J (extract a particle trajectory)
    J1 = find(rand(1) < cumsum(w(:,T)),1,'first');
    B = squeeze(particles1(J1,:,:,:));
    h = squeeze(particles2(J1,:,:));
    
end

end
%--------------------------------------------------------------------------
function [z1,z2,w] = cpf_as(X,N,q,A,beta,p,B,h,B_Mask)
% Conditional particle filter with ancestor sampling
% Input:
%   X - measurements
%   N - number of particles
%   q & p - process noise variances
%   B & h - conditioned particles - if not provided, un unconditional PF is run

conditioning = 1;
T = size(X,2);
m = size(X,1);
z1 = zeros(N,m,m,T); % Particles
z2 = zeros(N,m,T); % Particles
a1 = zeros(N, T); % Ancestor indices
a2 = zeros(N, T); % Ancestor indices
w = zeros(N, T); % Weights
z1(:,:,:,1) = 0; % Deterministic initial condition
z1(N,:,:,1) = B(:,:,1); % Set the 1st particle according to the conditioning
z2(:,:,1) = 0; % Deterministic initial condition
z2(N,:,1) = h(:,1)'; % Set the 1st particle according to the conditioning

for t = 1:T
    if(t==1)
        for i = 1:N
            z1(i,:,:,t) = sqrt(q).*randn(m,m);
            z2(i,:,t) = (sqrt(p).*randn(m,1))';
        end
        if(conditioning)
            z1(N,:,:,t) = B(:,:,t); % Set the Nth particle according to the conditioning
            z2(N,:,t) = h(:,t)';
        end
        % Compute importance weights
        LogLikelihoods = [];
        for i = 1:N
            %             E(:,i,t) = X(:,t)-squeeze(z1(i,:,:,t))*X(:,t); % E: m*N
            tmp = inv(eye(m)-squeeze(z1(i,:,:,t)) + 1e-5*eye(m));
            R = diag(squeeze(exp(z2(i,:,t))));
            Cov_X = tmp * R * tmp';
            Cov_X = Cov_X + 1e-5*eye(size(Cov_X,1));
            LogLikelihoods(i) = -0.5*(m*log(2*pi) + log(det(Cov_X)) + X(:,t)' * inv(Cov_X) * X(:,t));
        end
        LogLikelihoods = LogLikelihoods-max(LogLikelihoods)+700; % to aviod numerial issues
        for i = 1:N
            w(i,t) = 1/(sum(exp(LogLikelihoods-LogLikelihoods(i))));
        end
%         LogLikelihoods = LogLikelihoods-max(LogLikelihoods)+700; % to aviod computational issues
%         weights = exp(LogLikelihoods);
%         w(:,t) = weights/sum(weights); % Save the normalized weights

%         for i = 1:N
%             w(i,t) = 1/(sum(exp(LogLikelihoods-LogLikelihoods(i))));
%         end
    else
        ind1 = resampling(w(:,t-1));
        ind1 = ind1(randperm(N));
        ind2 = ind1;
        for i = 1:N
            zpred1(i,:,:) = A.*squeeze(z1(i,:,:,t-1));
            zpred2(i,:) = (beta.*squeeze(z2(i,:,t-1))')';
        end
        for i = 1:N
            z1(i,:,:,t) = squeeze(zpred1(ind1(i),:,:)) + sqrt(q).*randn(m,m);
            z2(i,:,t) = (squeeze(zpred2(ind2(i),:))' + sqrt(p).*randn(m,1))';
        end
        if(conditioning)
            z1(N,:,:,t) = B(:,:,t); % Set the N:th particle according to the conditioning
            z2(N,:,t) = h(:,t)';
            % Ancestor sampling
            for i = 1:N
                ll1(i) = llf1(squeeze(B(:,:,t)),squeeze(zpred1(i,:,:)),q,B_Mask);
                ll2(i) = llf2(squeeze(h(:,t)),squeeze(zpred2(i,:))',p);
            end
            ll1 = ll1 - max(ll1)+700; % to aviod numerical issues
            ll2 = ll2 - max(ll2)+700;
            for i = 1:N
                w1_as(i) = w(i,t-1)/(sum(w(:,t-1).*exp(ll1-ll1(i))'));
                w2_as(i) = w(i,t-1)/(sum(w(:,t-1).*exp(ll2-ll2(i))'));
            end
%             w1_as = w(:,t-1).*exp(ll1)';
%             w2_as = w(:,t-1).*exp(ll2)';
%             w1_as = w1_as/sum(w1_as);
%             w2_as = w2_as/sum(w2_as);
            
%             if(sum(w1_as)==0 | sum(w2_as)==0)
%                 display('warning');
%             end
%             for i = 1:N
%                 w1_as(i) = w(i,t-1)/(sum(w(:,t-1).*exp(ll1-ll1(i))'));
%                 w2_as(i) = w(i,t-1)/(sum(w(:,t-1).*exp(ll2-ll2(i))'));
%             end
            ind1(N) = find(rand(1) < cumsum(w1_as),1,'first');
            ind2(N) = find(rand(1) < cumsum(w2_as),1,'first');
        end
        % Store the ancestor indices
        a1(:,t) = ind1;
        a2(:,t) = ind2;
        
        % Compute importance weights
        LogLikelihoods = [];
        for i = 1:N
            %             E(:,i,t) = X(:,t)-squeeze(z1(i,:,:,t))*X(:,t); % E: m*N
            tmp = inv(eye(m)-squeeze(z1(i,:,:,t)) + 1e-5*eye(m));
            R = diag(squeeze(exp(z2(i,:,t))));
            Cov_X = tmp * R * tmp';
            Cov_X = Cov_X + 1e-5*eye(size(Cov_X,1));
            LogLikelihoods(i) = -0.5*(m*log(2*pi) + log(det(Cov_X)) + X(:,t)' * inv(Cov_X) * X(:,t));
        end
%         LogLikelihoods0 = LogLikelihoods;
        LogLikelihoods = LogLikelihoods-max(LogLikelihoods)+700; % to aviod numerical issues
        for i = 1:N
            w(i,t) = 1/(sum(exp(LogLikelihoods-LogLikelihoods(i))));
        end

%         weights = exp(LogLikelihoods);
%         w(:,t) = weights/sum(weights); % Save the normalized weights

%         if(sum(sum((weights)))==0)
%             display('warning')
%         end
%         if(sum(sum(isinf(weights)))>0)
%             display('warning')
%         end
%         if(sum(sum(isnan(weights)))>0)
%             display('warning')
%         end
%         for i = 1:N
%             w(i,t) = 1/(sum(exp(LogLikelihoods-LogLikelihoods(i))));
%         end
    end
end

% Generate the trajectories from ancestor indices
ind1 = a1(:,T);
ind2 = a2(:,T);
for t = T-1:-1:1
    z1(:,:,:,t) = z1(ind1,:,:,t);
    z2(:,:,t) = z2(ind2,:,t);
    ind1 = a1(ind1,t);
    ind2 = a2(ind2,t);
end
end


function S = update_M(S,particles1,particles2, w, gamma, X,B_Mask)
N = length(w); % number of particles
B = particles1;
h = particles2;
T = size(X,2);
m = size(X,1);
r = sum(sum(B_Mask));

Aa_tmp = zeros(r,r);
Ab_tmp = zeros(r,r);
qa_tmp = 0;
qb_tmp = zeros(r,r);

betaa_tmp = zeros(m,m);
betab_tmp = zeros(m,m);
pa_tmp = 0;
pb_tmp = zeros(m,m);

for i = 1:N
    for t = 1:T
        Bt = squeeze(B(i,:,:,t));
        ht = squeeze(h(i,:,t))';
        if(t>1)
            Bt1 = squeeze(B(i,:,:,t-1));
            ht1 = squeeze(h(i,:,t-1))';
            Bvt1 = matrix2vec(Bt1,B_Mask);
            Bvt = matrix2vec(Bt,B_Mask);
            qa_tmp = qa_tmp + w(i);
            qb_tmp = qb_tmp + w(i)*(Bvt-S.A*Bvt1)*(Bvt-S.A*Bvt1)';
            Aa_tmp = Aa_tmp + w(i)*Bvt*Bvt1';
            Ab_tmp = Ab_tmp + w(i)*Bvt1*Bvt1';
            betaa_tmp = betaa_tmp + w(i)*(ht*ht1');
            betab_tmp = betab_tmp + w(i)*(ht1*ht1');
            pa_tmp = pa_tmp + w(i);
            pb_tmp = pb_tmp + w(i)*(ht-S.beta*ht1)*(ht-S.beta*ht1)';
        end
    end
end
S.qa = (1-gamma)*S.qa + gamma*qa_tmp;
S.qb = diag(diag((1-gamma)*S.qb + gamma*qb_tmp));
S.q = S.qb/S.qa;

S.Aa = (1-gamma)*S.Aa + gamma*Aa_tmp;
S.Ab = (1-gamma)*S.Ab + gamma*Ab_tmp;
S.A = diag(diag(S.Aa*inv(S.Ab+1e-5*eye(size(S.Ab,1)))));

S.betaa = (1-gamma)*S.betaa + gamma*betaa_tmp;
S.betab = (1-gamma)*S.betab + gamma*betab_tmp;
S.beta = diag(diag(S.betaa*inv(S.betab+1e-5*eye(size(S.betab,1)))));

S.pa = (1-gamma)*S.pa + gamma*pa_tmp;
S.pb = diag(diag((1-gamma)*S.pb + gamma*pb_tmp));
S.p = S.pb/S.pa;
end

