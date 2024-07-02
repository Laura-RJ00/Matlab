%% Función para predecir el resultado de las clases con nuevos conjuntos de datos
% Unidad 3 - Sistemas de Información y Telemedicina

% Modelo entrenado previamente a partir de las 12 variables seleccionadas
% por métodos de filters y wrappers, y entrenado con un modelo de K-Vecinos
% con K=1, y con particiones de tipo KFold con k=15;

    %% Preparar el entorno para 
        clear all;
        close all;
        clc;

        modelo = load ('TrainedModelKNN12.mat'); %Cargar nuestro modelo predictivo 
        TrainedModel = modelo.MdlK1_12_KF;

        [x,y]=uigetfile('*.mat'); %Abrir el archivo Data_test.mat
        fileD=strcat(y,x);
        load (fileD);
        Datos=CORPUS_TEST;

    % Solucionar problema de valores perdidos
        missingLogic=ismissing(Datos);
        [posX,posY]=find(missingLogic==1);
        for c=1:1:length(posY)
            Datos(posX(c),posY(c))=mean(Datos(:,posY(c)),'omitnan');
        end

    % Solucionar problema de outliers
        Datos=filloutliers(Datos,'center','median');

        DatosLimpios=Datos(:,[1 3 4 8 10 11 14 15 17 18 20 21 22]);

    %% Evaluación del modelo con los datos de validación
        [predicciones,~,~] = predict(TrainedModel,DatosLimpios(:,1:end-1));
        sum=0;
        for i=1:1:length(predicciones)
            if predicciones(i)==DatosLimpios(i,end)
            sum=sum+1;
            end
        end
        errorPredicciones=(length(predicciones)-sum)/length(predicciones);
        precision=(1-errorPredicciones)*100;

        fprintf('El clasificador acertó  %.2f  de cada 100 casos' ,precision);
        
    % Matriz de confusión
        matrizConfusion=confusionmat(DatosLimpios(:,end),predicciones);
        confusionchart(matrizConfusion); title('Matriz de Confusión');

