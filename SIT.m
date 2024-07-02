%% Trabajo de la Unidad 3
% Desarrollo de un CDSS basado en datos biomédicos:  aplicación a diagnóstico 
% mediante cardiotocografías prenatales
% 
% *Materia*: Sistemas de Información y Telemedicina
% 
% *Alumnos:* 
% 
% - Carlos Daniel Sandoval Puentes
% - Irene López Ruiz
% - Alba Meneses Felipe
% - Laura Romero Jaque
% 
% 
%% ANÁLISIS EXPLORATORIO DE DATOS
    % Limpiar valores de ejecuciones pasadas
        clear all
        close all
        clc;

    % Cargar los datos
        load Data_train.mat
        data=CORPUS_TRAIN;
    % Nombres de las variables, para mostrar al graficar
        nombres=["   LB   ";"   AC   ";"   FM   ";"   UC   ";"   DL   ";"   DS   ";"   DP   ";"  ASTV  ";...
            "  MSTV  ";"  ALTV  ";"  MLTV  ";" Width  ";"  Min   ";"  Max   ";"  Nmax  ";" Nzeros ";...
            "  Mode  ";"  Mean  ";" Median ";"Variance";"Tendency"];
        nombres2=["   LB   ";"   FM   ";"   UC   ";"  ASTV  ";...
            "  MSTV  ";"  ALTV  ";"  MLTV  ";" Width  ";"  Min   ";"  Max   ";"  Nmax  ";...
            "  Mode  ";"  Mean  ";" Median ";"Variance";"Tendency"];

    % Solucionar problema de valores perdidos
        missingLogic=ismissing(data);
        [posX,posY]=find(missingLogic==1);
        for c=1:1:length(posY)
            data(posX(c),posY(c))=mean(data(:,posY(c)),'omitnan');
        end

    % Solucionar problema de outliers
        data=filloutliers(data,'center','median');

    %% División y graficación de los datos separados por clases
        % Normal=1 Sospechoso=2 Patológico=3
             % Clase 1
                Normal=data(data(:,end)==1,:);
             % Clase 2
                Sospechoso=data(data(:,end)==2,:);
             % Clase 3
                Patologico=data(data(:,end)==3,:);

    % Histograma
        dataR=data(:,1:end-1); % Datos sin etiqueta
        figure(1);
        for i=1:size(dataR,2)
            subplot(3,7,i);
            histogram(Normal(:,i),30);  
            hold on;
            histogram(Sospechoso(:,i),30); 
            hold on;
            histogram(Patologico(:,i),30);
            legend('Normal','Sospechoso','Patologico');
            title(nombres(i));
        end

    % Kernel density (Función densidad de probabilidad)
        figure(2);
        for i=1:size(dataR,2)
            subplot(3,7,i);
            [f,xi] = ksdensity(Normal(:,i));
            plot(xi,f);
            hold on;
            [f,xi] = ksdensity(Sospechoso(:,i));
            plot(xi,f);
            hold on;
            [f,xi] = ksdensity(Patologico(:,i));
            plot(xi,f);
            legend('Normal','Sospechoso','Patologico');
            title(nombres(i));
        end

    % QQ Plot Normal
        figure(3);
        for i=1:size(dataR,2)
            subplot(3,7,i);
            qqplot(Normal(:,i));
            title(nombres(i));
        end

    % QQ Plot Sospechoso
        figure(4);
        for i=1:size(dataR,2)
            subplot(3,7,i);
            qqplot(Sospechoso(:,i));
            title(nombres(i));
        end

    % QQ Plot Patologico
        figure(5);
        for i=1:size(dataR,2)
            subplot(3,7,i);
            qqplot(Patologico(:,i));
            title(nombres(i));
        end

    %% Ya se solucionaron los problemas de datos perdidos y OUTLIERS
    % Por lo que esta sección permanece comentada
    
        % %Box Plot Normal
        % figure;
        % for i=1:size(dataR,2)     
        %     subplot(5,5,i);
        %     boxplot(Normal(:,i));
        %     title(nombres(i));
        % end
        % 
        % %Box Plot Sospechoso
        % figure;
        % for i=1:size(dataR,2)     
        %     subplot(5,5,i);
        %     boxplot(Sospechoso(:,i));
        %     title(nombres(i));
        % end
        % 
        % %Box Plot Patologico
        % figure;
        % for i=1:size(dataR,2)     
        %     subplot(5,5,i);
        %     boxplot(Patologico(:,i));
        %     title(nombres(i));
        % end


    %% Matriz de gráficos de dispersión y correlación
        figure('Name','Plotmatriz');     
        plotmatrix(dataR); 
        title('Plotmatrix Normal');
        figure('Name','Correlación');
        rho = corr(dataR);
        imagesc(rho);
        colorbar;
        title('Correlación')

%% EXTRACCIÓN DE CARACTERÍSTICAS
    %% PCA  
        % Los datos aparecian mejor agrupados usando zscore
        dataZ=zscore(dataR);
        [COEFF, SCORE, LATENT,~, EXPLAINED] = pca(dataZ);

        % figure;
        % pareto(EXPLAINED);
        % figure
        % biplot(COEFF(:,1:2),'scores',SCORE(:,1:2),'varlabels',nombres);
        % 
        % limite1=length(Normal);
        % limite2=limite1+length(Sospechoso);
        % limite3=limite2+length(Patologico);
        % figure;
        % scatter(SCORE(1:limite1,1),SCORE(1:limite1,2),'g') 
        % hold on
        % scatter(SCORE(limite1+1:limite2,1),SCORE(limite1+1:limite2,2),'m') 
        % hold on
        % scatter(SCORE(limite2+1:limite3,1),SCORE(limite2+1:limite3,2),'r');
        % axis equal 
        % xlabel('1st Principal Component') 
        % ylabel('2nd Principal Component')

    %% Método indirecto FILTERS con las 21 variables
        %fscore
            data_indirecto=dataR;
            tags_indirecto=data(:,end);
            [out, rank_fscore] = fscore(data_indirecto,tags_indirecto); 
            out;
            rank_fscore;
        %relieff
            k=2; %k=nº de vecinos
            [ranks_relieff,weights] = relieff(data_indirecto,tags_indirecto,k); 
            ranks_relieff; 
            weights;

    %% Método directo WRAPPERS con las 21 variables
    % No se puede realizar debido a que hay variables de varianza=0
%         data_directo=dataR;
%         tags_directo=data(:,end);
%         fun = @(XT,yT,Xt,yt)sum((yt-classify(Xt,XT,yT,'quadratic'))~=0)/length(yt);     
%         inmodel_forward = sequentialfs(fun,data_directo,tags_directo, 'direction', 'forward'); 
%         inmodel_backward = sequentialfs(fun,data_directo,tags_directo,'direction', 'backward');
%         inmodel_forward
%         inmodel_backward


    %% NUEVA MATRIZ DE DATOS CON LAS VARIABLES ELEGIDAS
        %% Matriz de 16 variables (eliminadas con fscore)
            new_data16R=dataR(:,[1 3 4 8 9 10 11 12 13 14 15 17 18 19 20 21]); %sin etiquetas
            new_data16=[new_data16R,data(:,end)]; %con etiquetas 

        %% Matriz de 12 variables (eliminadas con wrappers(fordward))
            new_data12R=dataR(:,[1 3 4 8 10 11 14 15 17 18 20 21]);
            new_data12=[new_data12R,data(:,end)];

    %% PCA 16 variables
        new_16dataZ=zscore(new_data16R);
        [COEFF_new, SCORE_new, LATENT_new,~, EXPLAINED_new] = pca(new_16dataZ);

        % figure;
        pareto(EXPLAINED_new);
        figure
        biplot(COEFF_new(:,1:2),'scores',SCORE_new(:,1:2),'varlabels',nombres2);

        limite1=length(Normal);
        limite2=limite1+length(Sospechoso);
        limite3=limite2+length(Patologico);
        figure('Name','16 variables');
        scatter(SCORE_new(1:limite1,1),SCORE_new(1:limite1,2),'g') ;
        hold on
        scatter(SCORE_new(limite1+1:limite2,1),SCORE_new(limite1+1:limite2,2),'m');
        hold on
        scatter(SCORE_new(limite2+1:limite3,1),SCORE_new(limite2+1:limite3,2),'r');
        axis equal 
        xlabel('1st Principal Component') 
        ylabel('2nd Principal Component')


     %% Método directo WRAPPERS con 16 variables
         % Columnas 9,12,13 y 19 originales se eliminan por forward
                new_data_directo=new_data16R;
                new_tags_directo=new_data16(:,end);
                fun = @(XT,yT,Xt,yt)sum((yt-classify(Xt,XT,yT,'quadratic'))~=0)/length(yt);     
                new_inmodel_forward = sequentialfs(fun,new_data_directo,new_tags_directo, 'direction', 'forward'); 
        %       new_inmodel_backward = sequentialfs(fun,new_data_directo,new_tags_directo,'direction', 'backward');
                new_inmodel_forward;
        %       new_inmodel_backward;

    %% LDA con 16 variables
    % Se obtuvieron mejores resultados con PCA por lo que esta sección se
    % comentó para ahorrar coste computacional.
%     new_data_LDA=new_dataR;
%     new_tags_LDA=new_data(:,end);
%     [new_eigenmodel, new_W, new_l, new_ProjectedData] = flda(new_data_LDA, new_tags_LDA);
%     figure('Name','FLDA data*W');
%     newDataLDA=new_data_LDA*new_W;
%     newData=zscore(newData);
%     scatter(newDataLDA(1:limite1,1),newDataLDA(1:limite1,2),'r');
%     hold on
%     scatter(newDataLDA(limite1+1:limite2,1),newDataLDA(limite1+1:limite2,2),'b');
%     hold on
%     scatter(newDataLDA(limite2+1:limite3,1),newDataLDA(limite2+1:limite3,2),'g');
%     axis equal   
%     figure('Name','FLDA Projected Data');
%     scatter(new_ProjectedData(1:limite1,1),new_ProjectedData(1:limite1,2),'r');
%     hold on
%     scatter(new_ProjectedData(limite1+1:limite2,1),new_ProjectedData(limite1+1:limite2,2),'b');
%     hold on
%     scatter(new_ProjectedData(limite2+1:limite3,1),new_ProjectedData(limite2+1:limite3,2),'g');

%% CLASIFICACIÓN
   %% Modelos con 16 variables de fscore
         %Entrenar el modelo con cvpartition
            [row_data column_data]=size(new_data16);
            for count=1:1:100
            c=cvpartition(new_data16(:,end),'HoldOut',0.3);
         % Elementos de Entrenamiento
            idx = test(c)';
            aux1=find(idx==0);
            TrainData=new_data16(aux1,:);
         % Elementos de prueba
            aux1=find(idx==1);
            TestData=new_data16(aux1,:);
            
         %% Discriminante Lineal
            Mdllinear= fitcdiscr(TrainData(:,1:end-1),TrainData(:,end),'DiscrimType','linear');
            [labelL,scoreL,costL] = predict(Mdllinear,TestData(:,1:end-1));
            % ErrorL
            sum=0;
            for i=1:1:length(labelL)
                if labelL(i)==TestData(i,end)
                sum=sum+1;
                end
            end
            errorL16(count)=(length(labelL)-sum)/length(labelL); 
            
          %% Discriminante Cuadrático
%             MdlQ = fitcdiscr(TrainData(:,1:end-1),TrainData(:,end),'DiscrimType','quadratic');
%             [labelQ,scoreQ,costQ] = predict(MdlQ,TestData(:,1:end-1));
%             % ErrorQ
%             sum=0;
%             for i=1:1:length(labelQ)
%                 if labelQ(i)==TestData(i,end)
%                 sum=sum+1;
%                 end
%             end
%             errorQ16=(length(labelQ)-sum)/length(labelQ); 
            
          %% Modelo de K-Vecinos
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end));
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % ErrorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK16(count)=(length(labelK)-sum)/length(labelK); 
               
          %%  K-Vecinos con K=1
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % ErrorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK1_16(count)=(length(labelK)-sum)/length(labelK);
            
          %%  K-Vecinos con K=45
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',45);
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % ErrorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK45_16(count)=(length(labelK)-sum)/length(labelK);
               
          %%  K-Vecinos con K=round(sqrt(length(TrainData)))
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',round(sqrt(length(TrainData))));
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % ErrorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorKsqrt_16(count)=(length(labelK)-sum)/length(labelK);
        
            end
            
          %% ERRORES con 16 variables de fscore
                L16=mean(errorL16);
                KNN16_1=mean(errorK1_16);
                KNN16_45=mean(errorK45_16);
                KNN16_sqrt=mean(errorKsqrt_16);
               
   %% Modelos con las 16 variables de PCA 
        aux2(1:limite1,1)=1; aux2(limite1+1:limite2,1)=2; aux2(limite2+1:limite3,1)=3;
        SCOREM=SCORE_new; SCOREM(:,end+1)=aux2;
        for count=1:1:100
            c=cvpartition(SCOREM(:,end),'HoldOut',0.35);
            % Elementos de Entrenamiento
                idx = test(c)';
                aux1=find(idx==0);
                TrainData=SCOREM(aux1,:);
            % Elementos de prueba
                aux1=find(idx==1);
                TestData=SCOREM(aux1,:);
                
           %%  Discriminante lineal
               Mdllinear = fitcdiscr(TrainData(:,1:9),TrainData(:,end),'DiscrimType','linear');
               [labelL] = predict(Mdllinear,TestData(:,1:9));
               % ErrorL
               sum=0;
               for i=1:1:length(labelL)
                   if labelL(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorL_PCA(count)=(length(labelL)-sum)/length(labelL); 
               
           %%  Discriminante cuadrático
               MdlQ = fitcdiscr(TrainData(:,1:9),TrainData(:,end),'DiscrimType','Quadratic');
               [labelQ,scoreQ,costQ] = predict(MdlQ,TestData(:,1:9));
               % ErrorQ
               sum=0;
               for i=1:1:length(labelQ)
                   if labelQ(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorQ_PCA(count)=(length(labelQ)-sum)/length(labelQ);  
               
           %%  Modelo de K-vecinos    
               MdlK = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',1);
               [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:9));
               % ErrorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK1_PCA(count)=(length(labelK)-sum)/length(labelK);
               
           %% K-vecinos K=45
              MdlK = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',45);
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:9));
              % ErrorK
              sum=0;
             for i=1:1:length(labelK)
                 if labelK(i)==TestData(i,end)
                 sum=sum+1;
                 end
             end
             errorK45_PCA(count)=(length(labelK)-sum)/length(labelK);
             
           %%  K-vecinos con K=round(sqrt(length(TrainData)))
               MdlK = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',round(sqrt(length(TrainData))));
               [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:9));
               % Error K
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorKsqrt_PCA(count)=(length(labelK)-sum)/length(labelK);
        end
             
           %% ERRORES con 16 variables de PCA 
                L16PCA=mean(errorL_PCA);
                Q16PCA=mean(errorQ_PCA);
                Knn16_1PCA=mean(errorK1_PCA);
                Knn16_45PCA=mean(errorK45_PCA);
                Knn16_sqrtPCA=mean(errorKsqrt_PCA);
        
   %% Modelos con 12 variables de wrappers
         %Entrenar el modelo con cvpartition
            [row_data column_data]=size(new_data12);
            for count=1:1:100
            c=cvpartition(new_data12(:,end),'HoldOut',0.3);
         % Elementos de Entrenamiento
            idx = test(c)';
            aux1=find(idx==0);
            TrainData=new_data12(aux1,:);
         % Elementos de prueba
            aux1=find(idx==1);
            TestData=new_data12(aux1,:);
            
          %%  Discriminante Lineal
              Mdllinear= fitcdiscr(TrainData(:,1:end-1),TrainData(:,end),'DiscrimType','linear');
              [labelL,scoreL,costL] = predict(Mdllinear,TestData(:,1:end-1));
              % errorL
                 sum=0;
                 for i=1:1:length(labelL)
                     if labelL(i)==TestData(i,end)
                     sum=sum+1;
                     end
                 end
                 errorL12(count)=(length(labelL)-sum)/length(labelL); 
            
           %%  Discriminante Cuadrático
%             MdlQ = fitcdiscr(TrainData(:,1:end-1),TrainData(:,end),'DiscrimType','quadratic');
%             [labelQ,scoreQ,costQ] = predict(MdlQ,TestData(:,1:end-1));
%          % errorQ
%             sum=0;
%             for i=1:1:length(labelQ)
%                 if labelQ(i)==TestData(i,end)
%                 sum=sum+1;
%                 end
%             end
%             errorQ=(length(labelQ)-sum)/length(labelQ); 
            
          %%  Modelo de K-Vecinos
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end));
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % errorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK12(count)=(length(labelK)-sum)/length(labelK); 
               
          %%  K-Vecinos con K=1
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % errorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK1_12(count)=(length(labelK)-sum)/length(labelK);
            
          %%  K-Vecinos con K=45
              MdlK = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',45);
              [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:end-1));
              % errorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK45_12(count)=(length(labelK)-sum)/length(labelK);
               
          %%  K-Vecinos con K=round(sqrt(length(TrainData)))
              MdlK = fitcknn(TrainData,TrainData(:,end),'NumNeighbors',round(sqrt(length(TrainData))));
              [labelK,scoreK,costK] = predict(MdlK,TestData);
            % errorK
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorKsqrt_12(count)=(length(labelK)-sum)/length(labelK);
            end
            
          %% ERRORES con 12 variables Wrappers       
                L12=mean(errorL12);
                KNN12_1=mean(errorK1_12);
                KNN12_45=mean(errorK45_12);
                KNN12_sqrt=mean(errorKsqrt_12);
               
    %% Modelos con las 12 variables de PCA 
        new_12dataZ=zscore(new_data12R);
        [COEFF_new, SCORE_new, LATENT_new,~, EXPLAINED_new] = pca(new_12dataZ);
        aux2(1:limite1,1)=1; aux2(limite1+1:limite2,1)=2; aux2(limite2+1:limite3,1)=3;
        SCOREM=SCORE_new; SCOREM(:,end+1)=aux2;
        for count=1:1:100
        c=cvpartition(SCOREM(:,end),'HoldOut',0.35);
        % Elementos de Entrenamiento
            idx = test(c)';
            aux1=find(idx==0);
            TrainData=SCOREM(aux1,:);
        % Elementos de prueba
            aux1=find(idx==1);
            TestData=SCOREM(aux1,:);

           %%  Discriminante lineal
                Mdllinear = fitcdiscr(TrainData(:,1:9),TrainData(:,end),'DiscrimType','linear');
                [labelL] = predict(Mdllinear,TestData(:,1:9));
                sum=0;
                for i=1:1:length(labelL)
                    if labelL(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorL_PCA(count)=(length(labelL)-sum)/length(labelL); 
                
            %%  Discriminante cuadrático  
                MdlQ = fitcdiscr(TrainData(:,1:9),TrainData(:,end),'DiscrimType','Quadratic');
                [labelQ,scoreQ,costQ] = predict(MdlQ,TestData(:,1:9));
                sum=0;
                for i=1:1:length(labelQ)
                    if labelQ(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorQ_PCA(count)=(length(labelQ)-sum)/length(labelQ);   
                
            %%  Modelo con K-vecinos   
                MdlK = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',1);
                [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:9));
                sum=0;
                for i=1:1:length(labelK)
                    if labelK(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorK1_PCA(count)=(length(labelK)-sum)/length(labelK);
                
             %%  Modelo con K-vecinos=45 
                MdlK = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',45);
                [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:9));
                sum=0;
                for i=1:1:length(labelK)
                    if labelK(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorK45_PCA(count)=(length(labelK)-sum)/length(labelK);
                
             %%  Modelo con K-vecinos=round(sqrt(length(TrainData)))      
                 MdlK = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',round(sqrt(length(TrainData))));
                [labelK,scoreK,costK] = predict(MdlK,TestData(:,1:9));
                % errorK
                sum=0;
                for i=1:1:length(labelK)
                    if labelK(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorK45_PCA(count)=(length(labelK)-sum)/length(labelK);
        end
                
           %% ERRORES con 12 variables PCA
                L12PCA=mean(errorL_PCA);
                Q12PCA=mean(errorQ_PCA);
                Knn12_1PCA=mean(errorK1_PCA);   
                Knn12_45PCA=mean(errorK45_PCA); 
                Knn12_sqrtPCA=mean(errorKsqrt_PCA); 
        
    %% EVALUACIÓN
        % Los mejores métodos para este caso resultaron ser K-vecinos con k=1 tanto
        % para los métodos con PCA como para los métodos con las variables
        % originales estandarizadas.

        % Los métodos evaluados serán K-vecinos con K=1 y con 12 y 16 variables,
        % y PCA con 12 variables igual en K=1

     %% Entrenar el modelo con HoldOut al 35% (mejor resultado)
         for count=1:1:100
            c=cvpartition(new_data12(:,end),'HoldOut',0.35);
         %% K-vecinos con K=1 y 16 variables 
            % Elementos de Entrenamiento
                idx = test(c)';
                aux1=find(idx==0);
                TrainData=new_data16(aux1,:);
            % Elementos de prueba
                aux1=find(idx==1);
                TestData=new_data16(aux1,:);  
                MdlK1_16_HO = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
               [labelK,~,~] = predict(MdlK1_16_HO,TestData(:,1:end-1));
            % Error
                sum=0;  
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK1_16_HO(count)=(length(labelK)-sum)/length(labelK);

         %%  K-Vecinos con K=1 y 12 variables
             % Elementos de Entrenamiento
                idx = test(c)';
                aux1=find(idx==0);
                TrainData=new_data12(aux1,:);
             % Elementos de prueba
                aux1=find(idx==1);
                TestData=new_data12(aux1,:);
                MdlK1_12_HO = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
                [labelK,~,~] = predict(MdlK1_12_HO,TestData(:,1:end-1));
             % Error
                sum=0;
                for i=1:1:length(labelK)
                    if labelK(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorK1_12_HO(count)=(length(labelK)-sum)/length(labelK);

          %% PCA 12 variables y K=1
             c=cvpartition(SCOREM(:,end),'HoldOut',0.35);
            % Elementos de Entrenamiento
                idx = test(c)';
                aux1=find(idx==0);
                TrainData=SCOREM(aux1,:);
            % Elementos de prueba
                aux1=find(idx==1);
                TestData=SCOREM(aux1,:);
                MdlK1_PCA12_HO = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',1);
                [labelK,~,~] = predict(MdlK1_PCA12_HO,TestData(:,1:9));
            % Error
                sum=0;
                for i=1:1:length(labelK)
                    if labelK(i)==TestData(i,end)
                    sum=sum+1;
                    end
                end
                errorK1_PCA12_HO(count)=(length(labelK)-sum)/length(labelK);    
     end

    %%  Entrenar el modelo con K-Fold y K=15 (mejor resultado)
        for count=1:1:100
            c=cvpartition(new_data12(:,end),'Kfold',15);
            c2=cvpartition(SCOREM(:,end),'Kfold',15);
            for iter=1:1:10
                %% K-vecinos con K=1 y 16 variables       
                 % Elementos de Entrenamiento
                    idx = test(c,1)';
                    aux1=find(idx==0);
                    TrainData=new_data16(aux1,:);
                 % Elementos de prueba
                    aux1=find(idx==1);
                    TestData=new_data16(aux1,:);
                    MdlK1_16_KF = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
                   [labelK,~,~] = predict(MdlK1_16_KF,TestData(:,1:end-1));
                 % Error
                   sum=0;
                   for i=1:1:length(labelK)
                       if labelK(i)==TestData(i,end)
                       sum=sum+1;
                       end
                   end
                   errorK1_16_KF(iter,count)=(length(labelK)-sum)/length(labelK);

                %%  K-Vecinos con K=1 y 12 variables
                 % Elementos de Entrenamiento
                    idx = test(c,1)';
                    aux1=find(idx==0);
                    TrainData=new_data12(aux1,:);
                 % Elementos de prueba
                    aux1=find(idx==1);
                    TestData=new_data12(aux1,:);
                    MdlK1_12_KF = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
                   [labelK,~,~] = predict(MdlK1_12_KF,TestData(:,1:end-1));
                 % Error
                    sum=0;
                    for i=1:1:length(labelK)
                        if labelK(i)==TestData(i,end)
                        sum=sum+1;
                        end
                    end
                    errorK1_12_KF(iter,count)=(length(labelK)-sum)/length(labelK);

                %% PCA 12 variables y K=1
                    % Elementos de Entrenamiento
                        idx = test(c2,1)';
                        aux1=find(idx==0);
                        TrainData=SCOREM(aux1,:);
                    % Elementos de prueba
                        aux1=find(idx==1);
                        TestData=SCOREM(aux1,:);
                        MdlK1_PCA12_KF = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',1);
                       [labelK,~,~] = predict(MdlK1_PCA12_KF,TestData(:,1:9));
                    % Error
                        sum=0;
                        for i=1:1:length(labelK)
                            if labelK(i)==TestData(i,end)
                            sum=sum+1;
                            end
                        end
                        errorK1_PCA12_KF(iter,count)=(length(labelK)-sum)/length(labelK);
        end
        end

     %% Entrenar el modelo con LeaveOut
          for count=1:1:100
            c=cvpartition(new_data12(:,end),'LeaveOut');
            %% K-vecinos con K=1 y 16 variables
             % Elementos de Entrenamiento
                idx = test(c,1)';
                aux1=find(idx==0);
                TrainData=new_data16(aux1,:);
             % Elementos de prueba
                aux1=find(idx==1);
                TestData=new_data16(aux1,:);
                MdlK1_16_LO = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
                [labelK,~,~] = predict(MdlK1_16_LO,TestData(:,1:end-1));
             % Error
               sum=0;
               for i=1:1:length(labelK)
                   if labelK(i)==TestData(i,end)
                   sum=sum+1;
                   end
               end
               errorK1_16_LO(count)=(length(labelK)-sum)/length(labelK);

            %%  K-Vecinos con K=1 y 12 variables
                 % Elementos de Entrenamiento
                    idx = test(c,1)';
                    aux1=find(idx==0);
                    TrainData=new_data12(aux1,:);
                 % Elementos de prueba
                    aux1=find(idx==1);
                    TestData=new_data12(aux1,:);
                    MdlK1_12_LO = fitcknn(TrainData(:,1:end-1),TrainData(:,end),'NumNeighbors',1);
                   [labelK,~,~] = predict(MdlK1_12_LO,TestData(:,1:end-1));
                 % Error
                    sum=0;
                    for i=1:1:length(labelK)
                        if labelK(i)==TestData(i,end)
                        sum=sum+1;
                        end
                    end
                    errorK1_12_LO(count)=(length(labelK)-sum)/length(labelK);

            %% PCA 12 variables y K=1
               c=cvpartition(SCOREM(:,end),'LeaveOut');
                    % Elementos de Entrenamiento
                        idx = test(c,1)';
                        aux1=find(idx==0);
                        TrainData=SCOREM(aux1,:);
                    % Elementos de prueba
                        aux1=find(idx==1);
                        TestData=SCOREM(aux1,:);
                        MdlK1_PCA12_LO = fitcknn(TrainData(:,1:9),TrainData(:,end),'NumNeighbors',1);
                        [labelK,~,~] = predict(MdlK1_PCA12_LO,TestData(:,1:9));
                    % Erro
                        sum=0;
                        for i=1:1:length(labelK)
                            if labelK(i)==TestData(i,end)
                            sum=sum+1;
                            end
                        end
                        errorK1_PCA12_LO(count)=(length(labelK)-sum)/length(labelK);
                      end

      %% ERRORES DEFINITIVOS
          errorK1_16_HO=mean(errorK1_16_HO);
          errorK1_12_HO=mean(errorK1_12_HO);
          errorK1_PCA12_HO=mean(errorK1_PCA12_HO);
          errorK1_16_KF=mean(errorK1_16_KF(:));
          errorK1_12_KF=mean(errorK1_12_KF(:));
          errorK1_PCA12_KF=mean(errorK1_PCA12_KF(:));  
          errorK1_16_LO=mean(errorK1_16_LO);
          errorK1_12_LO=mean(errorK1_12_LO);
          errorK1_PCA12_LO=mean(errorK1_PCA12_LO);

    %   El error más pequeño se da con 12 variables y LeaveOut, sin embargo
    %   debido al tamaño de la base de datos y a que queremos evitar que el
    %   modelo quede sobre ajustado elegimos el segundo mejor error, que es
    %   algo más robusto para nuestro caso y consigue un buen porcentaje de
    %   error: K-Fold con K=15 y 12 variables
    % k=10 e=.1427 --- k=15 e=.1348 (500 ejecuciones) --- k=25 e=.1406 --- k=35 e=.1442
