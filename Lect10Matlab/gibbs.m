function gibbs(x,K,sigma,sigma0,alpha)
% gibbs(x,K,sigma,sigma0,alpha)
%
% theta_c   ~ N(0,sigma0^2 I )
% pi        ~ Dir(alpha/K)
% h         ~ pi
% x|h,theta ~ N(theta,sigma^2 I)
%

d = size(x,2);              % dimensionality of x
T = size(x,1);              % number of instances
h = ceil(rand(T,1)*K);             % hidden class assignment  
theta = 0.01*randn(K,d);         % means of gaussians
logpi = -log(K)*ones(K,1);  % mixing proportions
alphas = alpha/K*ones(K,1);

plotAll([],x,h,theta);
pause
MAXITER = 1000;
trace = zeros(1,MAXITER);
for it=1:MAXITER
    %%% Gibbs sampling for h
    lprob = zeros(K,1); % temp variably holding log prob
    for t=1:T
        for c=1:K
            lprob(c) = -d/2*log(2*pi) - 1/2*log(d*sigma) - 1/2*sum((x(t,:) - theta(c,:)).^2./sigma^2) + logpi(c);
        end
        lprob = lprob - max(lprob); lprob = lprob - log(sum(exp(lprob)));
        h(t) = find(rand(1) <= cumsum(exp(lprob)),1);
        if it<3 %% just for demo            
            plotAll(trace(1:it),x,h,theta)
        end
    end
    
    
    %%% Gibbs sampling for pi
    N = hist(h,1:K)';
    tpi = gamrnd(alphas+N,1); tpi = tpi/sum(tpi);
    logpi = log(tpi);
    
    %%% Gibbs sampling for theta
    for c=1:K
        m = sum(x(find(h==c),:))*sigma0^2/(sigma^2 + N(c)*sigma0^2);
        v = sigma^2*sigma0^2/(sigma^2 + N(c)*sigma0^2)*eye(d);
        s = sqrt(v);
        theta(c,:) = m + randn(1,d)*s;
    end
    trace(it) = lprobconf(x,h,theta,logpi,sigma,sigma0,alphas);
    plotAll(trace(1:it),x,h,theta)
end

function lprob=lprobconf(x,h,theta,logpi,sigma,sigma0,alphas)
K = size(theta,1);d = size(x,2);T = size(x,1);
lprob = 0;
for t=1:T
    lprob = lprob + -d/2*log(2*pi) - d/2*log(sigma) ...
                  - 1/2*sum((x(t,:) - theta(h(t),:)).^2./sigma^2) ...
                  + logpi(h(t));
end
N = hist(h,1:K);
lprob  = lprob + N*(alphas-1);
for c=1:K
    lprob = lprob + -d/2*log(2*pi) - d/2*log(sigma0) - 1/2*sum(theta(c,:).^2./sigma0);
end

function plotAll(trace,x,h,theta)
K = size(theta,1);
colors = {'r','g','b','k','c'};
subplot(2,1,1)
plot(trace(10:end));xlabel('iteration');ylabel('Log p');
subplot(2,1,2);
for c=1:K
    lst = find(h == c);
    plot(x(lst,1),x(lst,2),[colors{c} '.'],'MarkerSize',10);
    hold on
    plot(theta(c,1),theta(c,2),[colors{c} 'x'],'MarkerSize',20,'LineWidth',4);
end
hold off
drawnow;
