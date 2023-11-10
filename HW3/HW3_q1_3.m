% Part 3

%log-lin-fun = @(w) 1/(1+exp(-))

X = D10000_sample.';
Y = D10000_label.';

% a)
B1 = fitmnr(X, Y);
pred1=predict(B1,D20000_sample.');
p_error1=sum(D20000_label ~= pred1.')/20000

% b)
B2 = fitglm(X, Y,'quadratic','Distribution','Binomial');
pred2=predict(B2,D20000_sample.');
pred2=pred2 >= .5;
p_error2=sum(D20000_label ~= pred2.')/20000