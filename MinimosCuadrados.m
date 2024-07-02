function [X,error]=MinimosCuadrados(A,b)
%% PROGRAMA DE MÍNIMOS CUADRADOS
%Programa que devuelve el vector X que minimiza el residuo del sitema AX=b
%, es decir, la solución del problema de Minimos Cuadrados, y el error
%cometido. El cálculo se realiza via Ecuaciones Normales con la
%factorizacion de Cholesky.

%La factorización de Cholesky solo puede realizarse si la matriz es
%simétrica definida postitiva. Para conseguir esto, usaremos las ecuaciones
%normales para resolver el Problema de Mínimos Cuadrados:  A'Ax=A'b.
%Por tanto, realizamos la factorizacion de Cholesky sobre A'A
R=Cholesky(A'*A);
m=size(A'*A,1);
R_tras=R';

%Definimos el vector z=A'*b
z=A'*b;

%Como A'A=R'R (fact.Cholesky), podemos reescribir el sistema A'Ax=A'b
%como R'Rx=z, que es equivalente a RX=Y

%Resolvemos el sistema en dos subsistemas: primero, resolvemos R'Y=b para
%calcular el vector Y por sustitucion progresiva
Y(1)=z(1)/R_tras(1,1); 
for i=2:m
    Y(i)=(z(i)-R_tras(i,1:i-1)*Y(1:i-1))/R_tras(i,i);
end

%Ahora, resolvemos RX=Y por sustitución regresiva para calcular X
X(m)=Y(m)/R(m,m);
for j=m-1:-1:1
X(j)=(Y(j)-R(j,j+1:m)*X(j+1:m))/R(j,j);
end
X=X';

%Finalmente calculamos el error, definido como la norma euclidea
%del residuo r
r=b-A*X;
error=norm(r);
