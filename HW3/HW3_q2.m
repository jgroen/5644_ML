%Question 2
clear

T_mu = [0 0];
T_C = [1/16 0; 0 1/16];

location = mvnrnd(T_mu,T_C,1).';

angles=2*pi*rand(1,4);
radius = 1;
l_x = radius * cos(angles);
l_y = radius * sin(angles);
landmarks = [l_x ; l_y];

N_mu = 0;
N_C = 0.3^2;
distance=zeros(1,4);

measure = landmarks - location + normrnd(N_mu,N_C).';
for k=1:4
    distance(k)=norm(measure(:,k));
end

%Set up the grid
x = -2:.1:2;
y = -2:.1:2;
[X Y] = meshgrid(x,y);
for j=1:10
    limits(j)=0-1.7^(j);
end

%Set up the S probability
S = log(mvnpdf([X(:) Y(:)],T_mu,T_C));
S = reshape(S,size(X));

figure('Position',[100 100 500 500])
contour(X,Y,S,limits)
%zlim([-40 10])
hold on
scatter(location(1),location(2),100,">",'filled','b')
legend('','True Position')
%scatter(l_x,l_y,'filled','r')
ax=gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis square



%Set up the ri probilities
r=zeros(1,1681);
for k=1:4
    diff=[X(:) Y(:)]-landmarks(:,k).';
    for i=1:size(diff,1)
        dist(k,i) = norm(diff(i,:))-distance(1);
        r(:,i) = r(:,i)+log(normpdf(dist(k,i),N_mu,N_C));
    end
    R = reshape(r,size(X));
    figure('Position',[100 100 500 500])
    contour(X,Y,S+R,limits)
    %zlim([-40 10])
    hold on
    scatter(location(1),location(2),100,">",'filled','b')
    scatter(l_x(1:k),l_y(1:k),50,'filled','r')
    legend('','True Position','Landmark')
    ax=gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    axis square
end




