# Author: Fausto Sampaio
# Data: 14/07/2020

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

"""
    Classe para processar o dataset
"""
class ProcessarDados:
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.data = pd.read_csv(path_dataset, sep=',', header=0)

    # def normalization(self):
    #     x_mean = self.data_x().mean(axis=0)
    #     x_std = self.data_x().std(axis=0)
    #     return (self.data_x() - x_mean) / x_std, x_mean, x_std

    # retorna os atributos ou features dos dados
    def data_x(self):
        return self.data.values[0:, 0:10]

    # retorna as labels (classes) dos dados
    def data_y(self):
        return self.data.values[0:, 10]

    # Retorna os dados (misturados) particionados na sequencia em X_train, X_test, y_train, y_test
    # @perc_test percentual de dados para o teste
    def holdout(self, perc_test, seed):
        return train_test_split(self.data_x(), self.data_y(), test_size=perc_test, random_state=seed, shuffle=True)

    # Retorna k grupos de dados aleatorio do dataset (datax,datay)
    # Para cada grupo, temos a posicao 0: x_train, 1: y_train, 2: x_test, 3: y_test
    def kfolds(self, _datax, _datay, k):
        folds_data = []
        kf = KFold(n_splits=k, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(_datax):
            x_train, x_test = _datax[train_index], _datax[test_index]
            y_train, y_test = _datay[train_index], _datay[test_index]
            folds_data.append([x_train, y_train, x_test, y_test])

        return folds_data

