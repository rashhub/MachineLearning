function [qh,qh2,Z] = fwbw(x,K,logPh_1,logT,params)
L = size(x,2);
m_f = zeros(L,K);m_b = zeros(L,K);
qh = zeros(L,K);
qh2 = zeros(L-1,K,K);

for v=1:K
    m_f(1,v) = logPh_1(v) + logF(x(1),v,params);
end
m_b(L,:) = 0;


for l=2:L
    for v=1:K
        for vprev=1:K
            vec(vprev) = logT(vprev,v) + logF(x(l),v,params) + m_f(l-1,vprev);
        end
        m_f(l,v) = logsum(vec);
    end
end

for l=L-1:-1:1
    for v=1:K
        for vnext=1:K
            vec(vnext) = logT(v,vnext) + logF(x(l+1),vnext,params) + m_b(l+1,vnext);
        end
        m_b(l,v) = logsum(vec);
    end
end


Z = logsum(m_f(L,:));
assert(abs(logsum(m_f(L,:)) - logsum(m_b(1,:) + m_f(1,:)))<1e-6);


for l=1:L
    for v=1:K
        qh(l,v) = m_f(l,v) + m_b(l,v);
    end
    qh(l,:) = exp(qh(l,:) - logsum(qh(l,:)));
end

for l=1:L-1
    for v1=1:K
        for v2=1:K
            qh2(l,v1,v2) = m_f(l,v1) + logT(v1,v2) + logF(x(l+1),v2,params) + m_b(l+1,v2);
        end
    end
    qh2(l,:,:) = exp(qh2(l,:,:) - logsum(qh2(l,:,:)));
end



function lsum = logsum(vec)
vec = vec(:);
m = max(vec);
lsum = log(sum(exp(vec - m))) + m;

function lProb = logF(x,k,params)
iv = params(k).invvar;
m = params(k).mean;
lProb = -log(sqrt(2*pi)) + log(det(iv)) - 1/2*(x - m)'*iv*(x-m);
