% Part 2

prior1 = sum(D100_label) / 100;
prior0 = 1-prior1;

n=1;
nn=1;
for j=1:100
    if D100_label(j) == 1
        samp1(:,n)=D100_sample(:,j);
        n = n+1;
    else
        samp0(:,nn)=D100_sample(:,j);
        nn = nn+1;
    end
end

GM0 = fitgmdist(samp0.',2);
GM1 = fitgmdist(samp1.',1);

lratio = zeros(1,20000);
pred_label = zeros(2000,20000);
gamma = linspace(0,200,2000);

for i=1:20000
    lratio(i)=1/sqrt(det(GM1.Sigma)) * exp(-1/2*(D20000_sample(:,i)-GM1.mu.').' * inv(GM1.Sigma) * (D20000_sample(:,i)-GM1.mu.')) ...
        / (GM0.ComponentProportion(1)/(sqrt(det(GM0.Sigma(:,:,1)))) * exp(-1/2*(D20000_sample(:,i)-GM0.mu(:,1)).' * inv(GM0.Sigma(:,:,1)) * (D20000_sample(:,i)-GM0.mu(:,1))) ...
        + GM0.ComponentProportion(2)/(sqrt(det(GM0.Sigma(:,:,2)))) * exp(-1/2*(D20000_sample(:,i)-GM0.mu(:,2)).' * inv(GM0.Sigma(:,:,2)) * (D20000_sample(:,i)-GM0.mu(:,2))));
end

for k=1:2000
    pred_label(k,:) = lratio > gamma(k);
    pfa(k) = sum(pred_label(k,:)==1 & D20000_label==0) / sum(~D20000_label);
    pcd(k) = sum(pred_label(k,:)==1 & D20000_label==1) / sum(D20000_label);
    p_error(k) = prior0*pfa(k) + prior1*(1-pcd(k));
end

figure
plot(pfa,pcd);
title('Empirical ROC Curve')
xlabel('False positive rate')
ylabel('True positive rate')
legend('Minimum Probability of Error')
grid on

[M, I] = min(p_error)