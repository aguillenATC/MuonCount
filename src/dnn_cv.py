from data_reader import DataReader
import global_dirs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

import scipy

from model_manager import ModelManager

reader = DataReader
folds=global_dirs.folds

scaler_X=StandardScaler()
#scaler_y=StandardScaler()


mae = []
mse = []
r2 = []
dmean = []
dstd = []
rho = []
rho_pval = []
pearson = []
pearson_pval = []

for i in range(folds):
    train=pd.concat(reader.read_data(global_dirs.cross_val_data_path, formats=["hdf"], type="train", fold=str(i),verbose=False)[1].values())
    test=pd.concat(reader.read_data(global_dirs.cross_val_data_path, formats=["hdf"], type="test", fold=str(i), verbose=False)[1].values())

    manager=ModelManager()
    manager.assign_sets(train=train)
    tup = manager.create_mask(train.iloc[:, :-1], global_dirs.variable_selection[0],select=global_dirs.variable_selection[1])  # This tuple shouldn't take care about y_column index
    scalers = manager.preprocess_train(tup, scale_Y=False)

    dnn_model = manager.fit_dnn_regression([(9, 'relu'), (18, 'relu'), (56, 'relu'), (11, 'relu'), (10, 'relu')],
                                           epochs=300,
                                           batch_size=30,
                                           use_dropout=False)

    results=manager.predict_dnn_regression(test,tup)

    #X_train = train.loc[:, train.columns != train.columns[-1]]
    #X_test = test.loc[:, test.columns != test.columns[-1]]

    #var_selection = reader.create_mask(X_train, global_dirs.variable_selection[0], select=global_dirs.variable_selection[1])

    #X_train = X_train.loc[:, var_selection]
    #X_test = X_test.loc[:, var_selection]

    #y_train = train.loc[:, train.columns[-1]]
    #y_test = test.loc[:, test.columns[-1]]

    #X_train = scaler_X.fit_transform(X_train)
    #X_test = scaler_X.transform(X_test)

    #y_train= scaler_y.fit_transform(y_train.reshape(1,-1))
    #y_test = scaler_y.transform(y_test.reshape(1,-1))


    #[(90, 'relu'), (180, 'relu'), (560, 'relu'), (110, 'relu'), (100, 'relu'), (100, 'relu'), (100, 'relu')],
    #epochs = 500,
    #batch_size = 100,
    #save_dir = global_dirs.dnn_path + "model/",
    #use_dropout = False

    #net=fit_net(X_train,y_train,
    #            topology=[(9, 'relu'), (18, 'relu'), (56, 'relu'), (11, 'relu'), (10, 'relu')],
    #            epochs=500,
    #            batch_size=30,
    #            save_dir=global_dirs.dnn_path,
    #            use_dropout=False)
    #predictions = net.predict(X_test)
    #predictions = scaler_y.inverse_transform(net.predict(X_test))

    differences=results["differences"]

    fold_mae = results["mae"]#mean_absolute_error(y_test, predictions)
    fold_mse = results["mse"]#mean_squared_error(y_test, predictions)
    fold_r2 = results["r2"]#r2_score(y_test, predictions)
    fold_dmean = results["difference_mean"]#np.mean(differences)
    fold_dstd = results["difference_std"]#np.std(differences)
    fold_spearman_rho = results["spearman_rho"]
    fold_spearman_pval = results["spearman_pval"]#scipy.stats.spearmanr(y_test, predictions)
    fold_pearson_rho = results["pearson_rho"]
    fold_pearson_pval = results["pearson_pval"]#scipy.stats.pearsonr(y_test, predictions)

    print("Fold {}".format(i + 1))
    print("\tMAE: {}".format(fold_mae))
    print("\tMSE: {}".format(fold_mse))
    print("\tR2: {}".format(fold_r2))
    print("\tDiff mean: {}".format(fold_dmean))
    print("\tDiff std: {}".format(fold_dstd))
    print("\tSpearman rho: {}".format(fold_spearman_rho))
    print("\tSpearman p-value: {}".format(fold_spearman_pval))
    print("\tPearson rho: {}".format(fold_pearson_rho))
    print("\tPearson p-value: {}".format(fold_pearson_pval))

    mae.append(fold_mae)
    mse.append(fold_mse)
    r2.append(fold_r2)
    dmean.append(fold_dmean)
    dstd.append(fold_dstd)
    rho.append(fold_spearman_rho)
    rho_pval.append(fold_spearman_pval)
    pearson.append(fold_pearson_rho)
    pearson_pval.append(fold_pearson_pval)

results = pd.DataFrame.from_dict({
    'mae': mae,
    'mse': mse,
    'r2': r2,
    'dmean': dmean,
    'dstd': dstd,
    'spearman_rho': rho,
    'spearman_pval': rho_pval,
    'pearson_rho': pearson,
    'pearson_pval': pearson_pval
})

results.to_csv(global_dirs.dnn_path+"results/{}-fold-cross-validation-results.csv".format(folds),",",mode="w",header=True)
