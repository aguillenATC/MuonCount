from data_reader import DataReader
import global_dirs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pickle
import scipy


f = open(global_dirs.mlp_path+'best_params.pkl', 'rb')
best_params=pickle.load(f)
f.close()

print(best_params)

reader = DataReader
folds=5

scaler_X=StandardScaler()
#scaler_y=StandardScaler()

mlp=MLPRegressor(**best_params) #This is extracted from best_params

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

    X_train = train.loc[:, train.columns != train.columns[-1]]
    X_test = test.loc[:, test.columns != test.columns[-1]]

    var_selection = reader.create_mask(X_train,global_dirs.variable_selection[0], select=global_dirs.variable_selection[1])

    X_train = X_train.loc[:, var_selection]
    X_test = X_test.loc[:, var_selection]

    y_train = train.loc[:, train.columns[-1]]
    y_test = test.loc[:, test.columns[-1]]

    X_train = scaler_X.fit_transform(X_train,y_train)
    X_test = scaler_X.transform(X_test)

    mlp.fit(X_train,y_train)
    predictions=mlp.predict(X_test)

    differences=[r-p for r,p in zip(y_test,predictions)]

    fold_mae = mean_absolute_error(y_test, predictions)
    fold_mse = mean_squared_error(y_test, predictions)
    fold_r2 = r2_score(y_test, predictions)
    fold_dmean = np.mean(differences)
    fold_dstd = np.std(differences)
    fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_test, predictions)
    fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_test, predictions)

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

results.to_csv(global_dirs.mlp_path+"results/{}-fold-cross-validation-results.csv".format(folds),",",mode="w",header=True)