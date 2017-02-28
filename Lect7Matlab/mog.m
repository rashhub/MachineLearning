function [hstar] = mog(x,n,varargin)
figure(1);
clf
MAXITER = 100; TOL = 1e-6;
ll = zeros(MAXITER,1);

t = size(x,2);
dim = size(x,1);
mu = repmat(mean(x,2),[1 n])+0.01*randn(dim,n);
S = repmat(cov(x'),[1 1 n]);
lpi = -log(n)*ones(n,1);

invS = S;
for k=1:n
    invS(:,:,k) = inv(S(:,:,k));
end
q = 0.5*ones(t,n);

iter = 0;


while iter<MAXITER
    iter = iter +1;
    for k=1:n
        dx = x - repmat(mu(:,k),[1 t]);
        q(:,k) = -log(2*pi)*(dim/2) + 0.5*log(det(invS(:,:,k))) - 1/2*sum(dx'*invS(:,:,k).*dx',2) + lpi(k);
    end
   
    mm = max(q,[],2);
    q = q - repmat(mm,[1 n]);
    ll(iter) = sum(log(sum(exp(q),2)) + mm);
    q = q-repmat(log(sum(exp(q),2)),[1 n]);
    q = exp(q);
        
    for k=1:n
        mu(:,k) = sum((repmat(q(:,k)',[dim 1]).*x)./sum(q(:,k)),2);
        S(:,:,k) = ((repmat(q(:,k)',[dim 1]).*x)*x')./sum(q(:,k)) - mu(:,k)*mu(:,k)';
        invS(:,:,k) = inv(S(:,:,k));
        lpi(k) = log(sum(q(:,k))) - log(t);
    end
   
    if iter>1 && ll(iter) - ll(iter - 1)<TOL
        break
    end 
end

for i=1:t
    [~,hstar(i)] = max(q(i,:));
end


