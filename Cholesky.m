function R=Cholesky(A)
%Programa que devuelve la matriz R triangular superior de la factorizacion
%de Cholesky A=R'*R. Es necesario que la matriz A sea simétrica definida
%positiva para poder llevar a cabo la factorizacion.

%Creamos una matriz R de zeros de igual tamaño que la matriz A
n = size(A,1);
R = zeros(n);

%Realizamos el algoritmo de Cholesky para ir sustituyendo los ceros de R
%por los valores correctos
    for i=1:n
        R(i,i) = sqrt(A(i, i) - R(i, :)*R(i, :)');
        for j=(i + 1):n
        R(j, i) = (A(j, i) - R(i,:)*R(j ,:)')/R(i, i);
        end
    end

%El algoritmo nos da una matriz triangular inferior. Por haber definido que
%buscamos la triangular superior, definimos nuestra R como la traspuesta de
%la matriz obtenida en el bucle anterior
    R=R';