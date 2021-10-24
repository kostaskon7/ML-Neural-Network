P = [1 -1;1 -1];
W = [0.5 0.5];
figure;
plot(P(1,1),P(2,1),'r+');
hold on;
plot(P(1,2),P(2,2),'r+');
%Decision Boundary
x = -2 : .1 : 2;
y =(-W(1)*x )/W(2);
plot(x,y);
axis([-2 2 -2 2]);
title('Fig.3 Decision boundary for E10.4');
hold off;  
 
