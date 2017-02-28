rand('seed',2)
T = [0.975 0.025; 0.025 0.975];
mn = [-1 1];
n = 1000;
sd = sqrt([0.5 1]);
h = zeros(1,n);
x = zeros(1,n);
h(1) = 1;
for i=1:n
    x(i) = sd(h(i))*randn(1,1)+mn(h(i));
    ind = find(rand(1,1) < cumsum(T(h(i),:))); 
    if i<n
       h(i+1) = ind(1);
    end
end
figure(1)
clf
plot(find(h==1),x(find(h==1)),'b.','MarkerSize',16)
hold on
plot(find(h==2),x(find(h==2)),'r.','MarkerSize',16)
hold off
xlim([1 1000])
xlabel('l','FontSize',16);
ylabel('x_l','FontSize',16);
set(gca,'FontSize',16);
xlim([1 n]);
legend('h_l = 1','h_l = 2','FontSize',16)

savefig('../HMMCRFKalman/HMM1',gcf,'pdf')

%%
figure(2)
for k=1:2
    params(k).mean = mn(k); 
    params(k).invvar = sd(k)^-2;
end
hstar = vit(x,2,[0 -realmax],log(T),params)

clf
plot(find(hstar==1),x(find(hstar==1)),'b.','MarkerSize',16)
hold on
plot(find(hstar==2),x(find(hstar==2)),'r.','MarkerSize',16)
hold off
xlim([1 1000])
xlabel('l','FontSize',16);
ylabel('x_l','FontSize',16);
title('MAP from HMM on MoG','FontSize',16)
legend('h_l = 1','h_l = 2','FontSize',16)
set(gca,'FontSize',16);
savefig('../HMMCRFKalman/vit',gcf,'pdf')
%%

figure(3)
for k=1:2
    params(k).mean = mn(k); 
    params(k).invvar = sd(k)^-2;
end
hstar2 = vit(x,2,[0 -realmax],log(ones(2,2)),params)

clf
plot(find(hstar2==1),x(find(hstar2==1)),'b.','MarkerSize',16)
hold on
plot(find(hstar2==2),x(find(hstar2==2)),'r.','MarkerSize',16)
hold off
xlim([1 1000])
xlabel('l','FontSize',16);
ylabel('x_l','FontSize',16);
title('MAP from Mixture of Gaussians','FontSize',16)
legend('h_l = 1','h_l = 2','FontSize',16)
set(gca,'FontSize',16);
savefig('../HMMCRFKalman/mog',gcf,'pdf')
