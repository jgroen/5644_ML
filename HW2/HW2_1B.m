clear all;

[samples,true_label] = generateDataA1Q1(10000);

Mdl = fitcdiscr(samples.',true_label.');
wlda = [-0.370465788983876;-0.352518000215601];
pred_label = zeros(200,10000);
pred_value = wlda.'*samples;

tau = linspace(-5,5,200);

for k=1:200
    pred_label(k,:) = pred_value > tau(k);
    P11(k) = sum(pred_label(k,:).*true_label)/sum(true_label);
    P10(k) = sum(pred_label(k,:).*~true_label)/sum(~true_label);
    P01(k) = sum(~pred_label(k,:).*true_label)/sum(true_label);
end   

P_error = 0.65*P10+0.35*P01;
[M, I] = min(P_error)


figure
plot(P10,P11);
hold on;
scatter(P10(I),P11(I))
title('Empirical ROC Curve for LDA')
xlabel('False positive rate')
ylabel('True positive rate')
legend('LDA', 'Minimum Error Point')
