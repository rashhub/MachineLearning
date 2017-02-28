function [logT,logPh_1,params]=lrnhmm(x,K)
MAXITER = 100;
TOL = 1e-6;

D = size(x,1);
N = size(x,3);
L = size(x,2);
logT = -log(K)*ones(K,K);
logPh_1 = -log(K)*ones(K,1);

for k=1:K
    params(k).mean = mean(reshape(x,[D N*L]),2) + 0.01*randn(D,1);
    params(k).var = cov(reshape(x,[D N*L]));
    params(k).invvar = inv(params(k).var);
    params_new(k).mean = zeros(D,1);
    params_new(k).var = zeros(D,D);
    params_new(k).invvar = zeros(D,D);
end

iter = 0;
while iter<MAXITER
    iter = iter +1;
    
    T_new = zeros(K,K);
    Ph_1_new = zeros(K,1);
    
    for k=1:K
        params_new(k).var(:,:) = 0;
        params_new(k).mean(:) = 0;
    end
    totqh = zeros(K,1); logLik = 0;
    for t=1:N
        xt = x(:,:,t);
        [qh,qh2,Zt] = fwbw(xt,K,logPh_1,logT,params);
        T_new = T_new + squeeze(sum(qh2)); 
        Ph_1_new = Ph_1_new + qh(1,:)'; 
        for k=1:K
            for l=1:L
                params_new(k).mean = params_new(k).mean + qh(l,k)*xt(:,l);
                params_new(k).var = params_new(k).var + qh(l,k)*(xt(:,l)*xt(:,l)');
                totqh(k) = totqh(k)  + qh(l,k);
            end            
        end
        logLik = logLik + Zt;
    end
    
    T_new = T_new ./ repmat(sum(T_new,2),[1 K]);
    Ph_1_new = Ph_1_new ./ N;
    for k=1:K
        params_new(k).mean = params_new(k).mean / totqh(k);
        params_new(k).var = params_new(k).var / totqh(k);
        params_new(k).var = params_new(k).var - params_new(k).mean*params_new(k).mean';
        params_new(k).invvar = inv(params_new(k).var);
    end
       
    logT = log(T_new);
    logPh_1 = log(Ph_1_new);
    for k=1:K
         params(k).mean = params_new(k).mean;
         params(k).invvar = params_new(k).invvar;
    end
    
    ll(iter) = logLik;
    ll(1:iter)
    plot(ll(1:iter))

     
    if iter>1 && ll(iter) - ll(iter - 1)<TOL
        ll(iter) - ll(iter-1)
        break
    end 
end