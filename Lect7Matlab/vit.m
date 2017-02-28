function hMAP = maxprod(x,K,logPh_1,logT,params)
L = size(x,2);
m_f = zeros(L,K);m_b = zeros(L,K);

for v=1:K
    m_f(1,v) = logPh_1(v) + logF(x(1),v,params);
end
m_b(L,:) = 0;

for l=2:L
    for v=1:K
        for vprev=1:K
            vec(vprev) = logT(vprev,v) + logF(x(l),v,params) + m_f(l-1,vprev);
        end
        m_f(l,v) = max(vec);
    end
end

for l=L-1:-1:1
    for v=1:K
        for vnext=1:K
            vec(vnext) = logT(v,vnext) + logF(x(l+1),vnext,params) + m_b(l+1,vnext);
        end
        m_b(l,v) = max(vec);
    end
end
   
for l=1:L
    [maxVal,maxIndex] = max(m_b(l,:) + m_f(l,:));
    hMAP(l) = maxIndex;
end

function lProb = logF(x,k,params)
invvar = params(k).invvar;
mean = params(k).mean;
lProb = -log(sqrt(2*pi)) + log(det(invvar)) - 1/2*(x - mean)'*invvar*(x-mean);
