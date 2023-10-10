clear all;

[samples,true_label] = generateDataA1Q1(10000);

m1=[2; 2];
C1=[1 0; 0 1];
m01=[3; 0];
C01=[2 0; 0 1];
m02=[0; 3];
C02=[1 0; 0 2];
zm=[m01+m02];
zC=[C01+C02];

%lratio=2*sqrt(2)*exp(-1/2(samples(:,1)-m1).'*inv(C1)*(samples(:,1)-m1)) / ...
%    (exp(-1/2(samples(:,1)-m01).'*inv(C01)*(samples(:,1)-m01)) + ...
%    exp(-1/2(samples(:,1)-m02).'*inv(C02)*(samples(:,1)-m02)))


lratio = zeros(1,10000);

gamma = linspace(0,10,200);
pred_label = zeros(200,10000);
P11 = zeros(1,200);
P10 = zeros(1,200);

for i=1:10000
    lratio(i)=-1/2*(samples(:,i)-m1).'*inv(C1)*(samples(:,i)-m1) ...
    +1/2*(samples(:,i)-zm).'*inv(zC)*(samples(:,i)-zm);
end

for k=1:200
    pred_label(k,:) = lratio > log(gamma(k))-log(6);
    P11(k) = sum(pred_label(k,:).*true_label)/sum(true_label);
    P10(k) = sum(pred_label(k,:).*~true_label)/sum(~true_label);
    P01(k) = sum(~pred_label(k,:).*true_label)/sum(true_label);
end

gamma_min=log(.65/.35)-log(6);
pred_label_gm = lratio > gamma_min;
P11_gm=sum(pred_label_gm.*true_label)/sum(true_label);
P10_gm=sum(pred_label_gm.*~true_label)/sum(~true_label);


figure
plot(P10,P11);
hold on;
scatter(P10_gm,P11_gm);
title('Empirical ROC Curve')
xlabel('False positive rate')
ylabel('True positive rate')
legend('Minimum Expected Risk', 'Minimum Risk Operating Point')

figure
C = confusionmat(true_label,pred_label_gm);
confusionchart(C,[0 1])
title('Confusion Matrix')

P_error = .65*P10+.35*P01;
[M, I] = min(P_error)
log(gamma(I))-log(6)