t_0=find(data.Ti>=50,1); %Indice del vector Ti que corresponde con un tiempo de 50 ms.
n=length(data.Ti);       %Longitud del vector Ti
for k=1:8
    
    %Primera derivada
    for i=t_0:n  
        der(k,i)=(data.StateVars(i,k,29)-data.StateVars(i-1,k,29))/(data.Ti(i)-data.Ti(i-1));
    end
    
end

c=0; %Contador disparos 
for k=1:8
    for i=t_0:n
        %Condicion disparo potencial
        if der(k,i-1)<400 && der(k,i)>=400
            c=c+1;    %Contador de  disparos de UNA neurona
        end
        
    end
    Contador(1,k)=c; %Contador de disparos espontáneos de cada neurona
   
    c=0;             %Volvemos a iniciar el contador a 0 para la siguiente neurona
end
disp('                 Número de disparos              ')
disp('     N1    N2   N3    N4    N5    N6    N7    N8' )
disp(Contador)

%Vminimo
for k= 1:8  
    Vdendritas_min=-(data.StateVars(t_0:n,k,29));
    minimos=findpeaks(Vdendritas_min);
    Vmin(1,k)=-(min(minimos));
end
disp('                                        Vmin                                   ')
disp('      N1        N2        N3        N4        N5        N6        N7        N8' )
disp(Vmin) 

%Vpp
for k=1:8
    Vdendritas=data.StateVars(t_0:n,k,29);
    Vdendritas_min=-(data.StateVars(t_0:n,k,29));
    
    maximos=findpeaks(Vdendritas);      %Máximos de UNA neurona
    Vmax(1,k)=max(maximos);             %Máximos de TODAS las neuronas
    
    minimos=findpeaks(Vdendritas_min);  %Mínimos de UNA neurona
    Vmin(1,k)=-(min(minimos));          %Mínimos de TODAS las neuronas
    
    Vpp(1,k)=Vmax(1,k)-Vmin(1,k);       %Voltio pico-pico

end
disp('                                        Vpp                                   ')
disp('      N1        N2        N3        N4        N5        N6        N7        N8' )
disp(Vpp) 