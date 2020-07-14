from dados import ProcessarDados

# exemplo de uso da classe
procData = ProcessarDados("../dataset/norm_bin_10_FEATURES_M17_CM6b_TH199.csv")
X_train, X_test, y_train, y_test = procData.holdout(0.2)
data_folds = procData.kfolds(X_train, y_train, 5)
print(len((data_folds[0][0])))
print(len((data_folds[0][2])))
