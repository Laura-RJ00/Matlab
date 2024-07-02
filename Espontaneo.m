
t_0=find(data.Ti>=50,1);
for k=1:8
    
    %Segunda derivada
    for i=t_0:length(data.Ti)
        der2(k,i)=(der(k,i)-der(k,i-1))/(data.Ti(i)-data.Ti(i-1));
    end
     
end

c=0;     %Contador potenciales espont�neos
for k=1:8
    
    %Condici�n espont�neo o inducido
    for i=t_0:length(data.Ti)
        if der2(k,i-1)<1.5e4  && der2(k,i)>=1.5e4
            c=c+1; %Contador de disparos esp�nt�neos de UNA neurona
        end
    end
    Contador(1,k)=c;      %Contador de disparos espont�neos de cada neurona.
    Frecuencia(1,k)=Contador(1,k)/2500;  %Indice espontaneidad de cada neurona.
    
    c=0;      %Volvemos a iniciar el contador a 0 para la siguiente neurona
end

disp('            N�mero de disparos espont�neos       ')
disp('     N1    N2    N3    N4   N5    N6    N7    N8' )
disp(Contador) 
  

disp('                          Frecuencia de disparo espont�neo                    ')
disp('       N1        N2        N3        N4        N5       N6        N7       N8' )
disp(Frecuencia) 