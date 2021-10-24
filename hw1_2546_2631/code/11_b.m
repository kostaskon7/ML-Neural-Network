clear
[X,Y] = meshgrid(-3 : .1 : 3);
F = 1 - 2 * (X + Y) + (X + Y).^2;
surf(X,Y,F)
title('Fig.1 Surface plot of Mean Square Error');
figure;
contour(X,Y,F)
title('Fig.2 Trajectory for different initial weights');
hold on;

%Initialize data
P = [1 -1;1 -1];
T = [1 -1];
alfa = 0.2;
W1 = [0;0];
W2 = [1;1];
for k = 1 : 2
   if (k == 1)
      W = W1;
   else
      W = W2;
   end
   plot(W(1), W(2),'r*')
   text(-0.3,-0.3,'W_0 =(0,0)');
   text(1,1.2,'W_0 =(1,1)');
   
	%Train the network
	for step = 1 : 20
   	for i = 1 : 2
      	a = purelin(W' * P(:,i));
      	e = T(i) - a;
      	W = W + 2 * alfa * e * P(:,i);
         if (k == 1)
            plot(W(1), W(2),'k.')
            W1 = W;
         else 
            plot(W(1), W(2),'b+')
            W2 = W;
         end
   	end
	end
end
W1
W2
hold off;  
