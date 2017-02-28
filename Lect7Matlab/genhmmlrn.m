rand('seed',2)
T = [0.975 0.025; 0.025 0.975];
mn = [-1 1];
L = 100;N=10;
sd = sqrt([0.5 1]);
h = zeros(1,L);
x = zeros(1,L,N);
h(1) = 1;
for t=1:N
    for i=1:L
        x(:,i,t) = sd(h(i))*randn(1,1)+mn(h(i));
        ind = find(rand(1,1) < cumsum(T(h(i),:)));
        if i<L
            h(i+1) = ind(1);
        end
    end
end
for k=1:2
    params(k).mean = mn(k); 
    params(k).invvar = sd(k)^-2;
end