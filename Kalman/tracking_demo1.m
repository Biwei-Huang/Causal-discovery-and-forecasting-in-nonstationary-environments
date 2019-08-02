% Make a point move in the 2D plane
% State = (x y xdot ydot). We only observe (x y).

% This code was used to generate Figure 15.9 of "Artificial Intelligence: a Modern Approach",
% Russell and Norvig, 2nd edition, Prentice Hall, 2003.

% X(t+1) = F X(t) + noise(Q)
% Y(t) = H X(t) + noise(R)
addpath('~/Dropbox/MATLAB/disagrregation/KalmanAll/KPMstats');
addpath('~/Dropbox/MATLAB/disagrregation/KalmanAll/KPMtools');
ss = 6; % state size
os = 2; % observation size
F = zeros(ss);
F(5,1) = 1;
F(6,2) = 1;
I = eye(os);
A = [0.5 0; 0.3 0.4];
H = [I I+A A];
% Q = 0.1*eye(ss);
% Q(ss-1,ss-1) = 0;
% Q(ss,ss) = 0;
Q = diag([0.1 0.01 0.1 0.1 0 0]);
R = 0*eye(os);
initx = zeros(ss,1);
% initV = diag([0.1 0.01 0.1 0.1 0.01 0.01]);
initV = 10*eye(ss);
seed = 1;
rand('state', seed);
randn('state', seed);
T = 15;
[x,y] = sample_lds(F, H, Q, R, initx, initV, T);
model = 1:T;
Q = repmat(Q, [1,1,T]);
[xfilt, Vfilt, VVfilt, loglik] = kalman_filter(y, F, H, Q, R, initx, initV,'model',model);
[xsmooth, Vsmooth, VVsmooth] = kalman_smoother(y, F, H, Q, R, initx, initV,'model',model);

dfilt = x([1 2],:) - xfilt([1 2],:);
mse_filt = sqrt(sum(sum(dfilt.^2)))

dsmooth = x([1 2],:) - xsmooth([1 2],:);
mse_smooth = sqrt(sum(sum(dsmooth.^2)))


figure(1)
clf
%subplot(2,1,1)
hold on
plot(x(1,:), x(2,:), 'ks-');
plot(y(1,:), y(2,:), 'g*');
plot(xfilt(1,:), xfilt(2,:), 'rx:');
for t=1:T, plotgauss2d(xfilt(1:2,t), Vfilt(1:2, 1:2, t)); end
hold off
legend('true', 'observed', 'filtered', 3)
xlabel('x')
ylabel('y')



% 3x3 inches
set(gcf,'units','inches');
set(gcf,'PaperPosition',[0 0 3 3])  
%print(gcf,'-depsc','/home/eecs/murphyk/public_html/Bayes/Figures/aima_filtered.eps');
%print(gcf,'-djpeg','-r100', '/home/eecs/murphyk/public_html/Bayes/Figures/aima_filtered.jpg');


figure(2)
%subplot(2,1,2)
hold on
plot(x(1,:), x(2,:), 'ks-');
plot(y(1,:), y(2,:), 'g*');
plot(xsmooth(1,:), xsmooth(2,:), 'rx:');
for t=1:T, plotgauss2d(xsmooth(1:2,t), Vsmooth(1:2, 1:2, t)); end
hold off
legend('true', 'observed', 'smoothed', 3)
xlabel('x')
ylabel('y')


% 3x3 inches
set(gcf,'units','inches');
set(gcf,'PaperPosition',[0 0 3 3])  
%print(gcf,'-djpeg','-r100', '/home/eecs/murphyk/public_html/Bayes/Figures/aima_smoothed.jpg');
%print(gcf,'-depsc','/home/eecs/murphyk/public_html/Bayes/Figures/aima_smoothed.eps');
