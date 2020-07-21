" Importações das bibliotecas"
import pandas as pd
import numpy as np
# Algoritmo KNN
from sklearn.neighbors import KNeighborsClassifier
# Uma das formas mais completas para Validação Cruzada com K-Fold e para aplicação dos dados de treinamento e teste, além da Verificação dos Scores da Acurácia, Recall, F1_Score, Precision e Análise da Matriz de Confusão"
from sklearn.model_selection import StratifiedKFold
# Metricas para avaliação
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
# GridSearch
from sklearn.model_selection import GridSearchCV

"Importando dados CSV"
# Base de dados do datagrid
base = pd.read_csv(r'C:\Users\Eduardo Valente\Google Drive\_Doutorado 2019_\2020\Aulas\CKP8277 - APRENDIZAGEM AUTOMATICA (2020.1 - T01)\Trabalho Final\_codigos para artigo final\norm_bin_10_FEATURES_M17_CM6b_TH199.csv')
               
# Realizando a divsão do datagrid em previsores e classe              
previsores = base.iloc[:, 0:10].values
classe = base.iloc[:, 10].values

#Escolhendo o melhor parâmetro do 'n_neighbors' para a aexcolha do K-Vizinho com GridSearch
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1,25)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(previsores, classe)
knn_gscv.best_params_
param_knn_gscv = int(knn_gscv.best_params_['n_neighbors']) 
#knn_gscv.best_score_

# Definindo épocas para captura dos resultados dos K-Folds para Validação Cruzada
epocas = 50;

# Divisão dos K_Folds
qtd_folds = 5

#Variáveis com array para armazenar os resultados finais das métricas
resultado_final_acuracia = []
resultado_final_fscore = []
resultado_final_precisao = []
resultado_final_recall = []
resultado_final_matriz = []

for i in range(epocas):
    
    # Aplicando o K-Fold para aplicação da Validação Cruzada sobre os dados de Testes e Ttreinamento
    kfold = StratifiedKFold(n_splits=qtd_folds, shuffle=True, random_state = None)
    
    #Variáveis com array para armazenar as das métricas de cada rodada
    resultados_acuracia = []
    resultados_fscore = []
    resultados_precisao = []
    resultados_recall = []
    resultados_matriz = []
    
    # Início do FOR da execução das 5 rodadas do K-Fold da validação cruzada
    for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
        #Classificando com KNN
        classificador = KNeighborsClassifier(n_neighbors=param_knn_gscv, metric='minkowski', p = 2)
        #Treino
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        #Realizando Testes
        previsoes = classificador.predict(previsores[indice_teste])
        #Realizando Métricas
        acuracia = accuracy_score(classe[indice_teste], previsoes)
        fscore = f1_score(classe[indice_teste], previsoes)
        precisao = precision_score(classe[indice_teste], previsoes)
        Recall = recall_score(classe[indice_teste], previsoes)
      
        #Guardando Resultados das métricas
        resultados_acuracia.append(acuracia)
        resultados_fscore.append(fscore)
        resultados_precisao.append(precisao)
        resultados_recall.append(Recall)
        resultados_matriz.append(confusion_matrix(classe[indice_teste], previsoes))

        # FIM do FOR da execução das 10 rodadas do K-FOLD da validação cruzada
    
    #Resultados
    resultados_acuracia = np.asarray(resultados_acuracia)
    resultados_fscore = np.asarray(resultados_fscore)
    resultados_precisao = np.asarray(resultados_precisao)
    resultados_recall = np.asarray(resultados_recall)

    #Média dos resultados Fimaos
    media_acuracia = resultados_acuracia.mean()
    media_fscore = resultados_fscore.mean()
    media_precisao = resultados_precisao.mean()
    media_recall = resultados_recall.mean()
    media_matriz = np.mean(resultados_matriz, axis = 0)
    
    resultado_final_acuracia.append(media_acuracia)
    resultado_final_fscore.append(media_fscore)
    resultado_final_precisao.append(media_precisao)
    resultado_final_recall.append(media_recall)
    resultado_final_matriz.append(media_matriz)
    # FIM da execução do FOR 50 vezes para capturar os dados 'sementes' de resultados 
    
#Resultados finais
media_final_acuracia = np.asarray(resultado_final_acuracia)
media_final_fscore = np.asarray(resultado_final_fscore)
media_final_precisao = np.asarray(resultado_final_precisao)
media_final_recall = np.asarray(resultado_final_recall)

media_final_acuracia.mean()
media_final_fscore.mean()
media_final_precisao.mean()
media_final_recall.mean()
media_final_matriz = np.mean(resultado_final_matriz, axis = 0)

media_final_acuracia.std()
media_final_fscore.std()
media_final_precisao.std()
media_final_recall.std()


print ("\t Accuracy \t|\t Precision \t|\t Recall \t|\t F1-Score")
print ("      %.2f +- %.3f" % (media_final_acuracia.mean(), media_final_acuracia.std()),end='       ')
print ("      %.2f +- %.3f" % (media_final_fscore.mean(), media_final_fscore.std()),end='       ')
print ("      %.2f +- %.3f" % (media_final_precisao.mean(), media_final_precisao.std()),end='       ')
print ("      %.2f +- %.3f" % (media_final_recall.mean(), media_final_recall.std()),end='       ')
print ("====================================================================================================")

# Acurácia: indica uma performance geral do modelo. Dentre todas as classificações, quantas o modelo classificou corretamente.
# Precisão: dentre todas as classificações de classe Positivo que o modelo fez, quantas estão corretas.
# Recall/Revocação/Sensibilidade: Dentre todas as situações de classe Positivo como valor esperado, quantas estão corretas.
# F1-Score: Média harmônica entre precisão e recall.

#for i in range(resultado_final_acuracia.size): print(str(resultado_final_acuracia[i]).replace('.', ','))

