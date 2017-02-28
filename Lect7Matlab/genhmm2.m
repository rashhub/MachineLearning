rand('seed',4);
T = [0.95 0.05 0; 0 0 1; 0.1 0.9 0];
f = [1/4*ones(1,4); 0 1 0 0; 0 0 0 1];
x = zeros(1,n);
h = zeros(1,n);
n = 100;
h(1) = 1;
for i=1:n
    r = rand(1,1);
    ind = find(r<cumsum(f(h(i),:)));
    x(i) = ind(1);
    r = rand(1,1);
    ind = find(r<cumsum(T(h(i),:)));
    if i<n
        h(i+1) = ind(1);
    end
end
curr = 1;
str = ['\textcolor{' color{curr} '}{'];
map = 'ACTG';
color{1} = 'blue';
color{2} = 'red';
color{3} = 'green';
for i=1:100
    
    if h(i) ~= curr
        str = [str sprintf('}\\textcolor{%s}{',color{h(i)})];
        curr = h(i);
    end
    str = [str map(x(i))];
    if mod(i,40)==0 && i<length(x)
        str = [str '}'];
        str = [str 10 '\vskip 5pt' 10 '\textcolor{' color{curr} '}{' ];
    end
end
str = [str '}']
fid = fopen('../HMMCRFKalman/hmm2.tex','w+');
fprintf(fid,'%s',str);
fclose(fid);
