% Part 1
m1=[3; 2];
C1=[2 0; 0 2];
m01=[5; 0];
C01=[4 0; 0 2];
m02=[0; 4];
C02=[1 0; 0 3];


lratio = zeros(1,20000);
pred_label = zeros(2000,20000);
gamma = linspace(0,200,2000);

for i=1:20000
    lratio(i)=1/sqrt(det(C1)) * exp(-1/2*(D20000_sample(:,i)-m1).' * inv(C1) * (D20000_sample(:,i)-m1)) ...
        / (1/(2*sqrt(det(C01))) * exp(-1/2*(D20000_sample(:,i)-m01).' * inv(C01) * (D20000_sample(:,i)-m01)) ...
        + 1/(2*sqrt(det(C02))) * exp(-1/2*(D20000_sample(:,i)-m02).' * inv(C02) * (D20000_sample(:,i)-m02)));
end

pred_label_theory = lratio > 3/2;
pfa_theory = sum(pred_label_theory==1 & D20000_label==0) / sum(~D20000_label);
pcd_theory = sum(pred_label_theory==1 & D20000_label==1) / sum(D20000_label);
p_error_theory = 0.6*pfa_theory + 0.4*(1-pcd_theory)

for k=1:2000
    pred_label(k,:) = lratio > gamma(k);
    pfa(k) = sum(pred_label(k,:)==1 & D20000_label==0) / sum(~D20000_label);
    pcd(k) = sum(pred_label(k,:)==1 & D20000_label==1) / sum(D20000_label);
    p_error(k) = 0.6*pfa(k) + 0.4*(1-pcd(k));
end

figure
plot(pfa,pcd);
hold on;
scatter(pfa_theory,pcd_theory);
title('Empirical ROC Curve')
xlabel('False positive rate')
ylabel('True positive rate')
legend('Minimum Probability of Error', 'Min-P(error) classifier operating point')

[M, I] = min(p_error)

% Plot the points 
boundary=abs(lratio-3/2)<.1;

n=1;
nn=1;
nnn=1;
for j=1:20000
    if D20000_label(j) == 1
        samp1(:,n)=D20000_sample(:,j);
        n = n+1;
    else
        samp0(:,nn)=D20000_sample(:,j);
        nn = nn+1;
    end
    if boundary(j) == 1
        bound_line(:,nnn)=D20000_sample(:,j);
        nnn = nnn+1;
    end
end

figure
scatter(samp0(1,:),samp0(2,:),'.','b')
hold on
scatter(samp1(1,:),samp1(2,:),'.','r')
scatter(bound_line(1,:),bound_line(2,:),'g','filled')
axis tight
grid on
legend('Class 0', 'Class 1','Decision Boundry')
title('20K Data set')

