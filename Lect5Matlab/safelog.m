function lp=safelog(p)
lp = log(p);

lp(find(p==0)) = -realmax;