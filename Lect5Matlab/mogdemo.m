function mogdemo
runmogdemo
 
function mog(x,n,varargin)
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
showState(x,q(:,1),mu,S,iter,ll(1:iter));

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
   
     showState(x,q(:,1),mu,S,iter,ll(1:iter));
    if iter>1 && ll(iter) - ll(iter - 1)<TOL
        break
    end 
end


%%%
function runmogdemo
A1 = [1/sqrt(2) -1/sqrt(2); 1/sqrt(2) 1/sqrt(2)];
D1 = diag([6 1]);
m1 = [1;5];

A2 = [1/sqrt(2) -1/sqrt(2); 1/sqrt(2) 1/sqrt(2)];
D2 = diag([2 3]);
m2 = [-3;15];

t1 = 200; t2 = 400;
cls1 = A1*D1*randn(2,t1) + repmat(m1,[1 t1]);

cls2 = A2*D2*randn(2,t2) + repmat(m2,[1 t2]);
 x = [cls1,cls2];
for i=1:10
 mog(x,2)
 pause(10)
end
function showState(x,q,ms,cs,iter,ll)
colors={'b','r'};
subplot(1,2,1)
for i=1:length(x)
    if (q(i) >= 0.5)
        color = min(2*[1.0-q(i) 1.0-q(i) 0.5],1.0);
    else
        color = min(2*[0.5 q(i) q(i)],1.0);
    end
    plot(x(1,i),x(2,i),'o','MarkerSize',3,'LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor',color);
    hold on
end
for i=1:size(ms,2)    
    plotGaussian(ms(:,i),cs(:,:,i),colors{i});    
    plot(ms(1,i),ms(2,i),[colors{i} 'x'],'LineWidth',10,'MarkerSize',5);
end
hold off
title(['Iteration: ' num2str(iter)],'FontSize',20)
subplot(1,2,2)
plot(ll,'LineWidth',4)
xlabel('Iteration','FontSize',16);
ylabel('Log likelihood','FontSize',16);

drawnow
function plotGaussian(m,c,color)
[V,D] = eig(c);
[d,ii] = sort(diag(D),'ascend');
D = diag(d);
V = V(:,ii);
t = [0:0.01:2*pi];
x = [cos(t);sin(t)];
W = V*sqrt(D)*2;
x = W*x + repmat(m,[1 length(t)]);
plot(x(1,:),x(2,:),'Color',color,'LineWidth',3);