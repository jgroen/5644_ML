clear all;

[samples,true_label] = generateDataQ2(10000);

Mdl = fitcnb(samples',true_label');

pred_label = predict(Mdl,samples')';

figure
C = confusionmat(true_label,pred_label);
confusionchart(C,[1 2 3],'Normalization','row-normalized')
title('Confusion Matrix')

correct = true_label == pred_label;

figure
for l = 0:2
    %indl = find(true_label==l);
    indl_c = find(true_label.*correct==l);
    indl_f = find(true_label.*~correct==l);

    if l == 0
        scatter3(samples(1,indl_f),samples(2,indl_f),samples(3,indl_f),'o','r'), hold on
        scatter3(samples(1,indl_c),samples(2,indl_c),samples(3,indl_c),'o','g'), hold on
    elseif l == 1
        scatter3(samples(1,indl_f),samples(2,indl_f),samples(3,indl_f),'+','r'), hold on
        scatter3(samples(1,indl_c),samples(2,indl_c),samples(3,indl_c),'+','g'), hold on
    elseif l == 2
        scatter3(samples(1,indl_f),samples(2,indl_f),samples(3,indl_f),'square','r'), hold on
        scatter3(samples(1,indl_c),samples(2,indl_c),samples(3,indl_c),'square','g')
    end
    axis equal
    title('3D representation of data')
    legend('','Class 1','','Class 2','','Class 3')
end