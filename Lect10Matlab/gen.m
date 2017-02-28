rand('seed',1);
randn('seed',7);
sigma0 = 3;
sigma = 1;
d = 2;
K = 5;
thetas = sigma0*randn(K,d);
ps = [0.15 0.15 0.2 0.2 0.3];
T = 400;
cts = mnrnd(T,ps);
smpls = [];
h = [];
colors = {'r','g','b','k','c'};
for i=1:K
    nsmpls = repmat(thetas(i,:),[cts(i) 1]) + sigma*randn(cts(i),d);
    plot(nsmpls(:,1),nsmpls(:,2),[colors{i} '.'])
    hold on
    smpls = [smpls;nsmpls];
    h = [h;i*ones(cts(i),1)];
end
hold off

clf
pause
gibbs(smpls,K,sigma,sigma0,1)
