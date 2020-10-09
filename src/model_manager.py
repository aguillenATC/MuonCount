from data_reader import DataReader
#import global_dirs

import os

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

from xgboost.sklearn import XGBRegressor

import scipy



#import GPy

class ModelManager(DataReader):

    def __init__(self):
        super().__init__()
        self.X_train=None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.scaler_X = None
        self.scaler_y = None

        self.linear_reg_model=None
        self.poly_reg_model = None
        self.dt_reg_model = None
        self.rf_reg_model = None
        self.boost_reg_model = None
        self.xgboost_reg_model = None
        #self.gproc_reg_model = None
        self.svm_reg_model = None
        self.mlp_reg_model = None
        self.dnn_reg_model = None

        # Parameters GridSearch for models
        self.gs_dt = None
        self.gs_rf = None
        self.gs_boost = None
        self.gs_xgboost = None
        #self.gs_gproc= None
        self.gs_svm = None
        self.gs_mlp = None

        #Parameters for poly regression
        self.poly_X = None
        self.poly_y = None

        self.y_column=None

        self.is_y_scaled=False

    def assign_sets(self, train, val=None, y_column=-1):
        self.X_train = pd.DataFrame(train.loc[:, train.columns!=train.columns[y_column]])
        self.y_train = pd.Series(train.loc[:, train.columns[y_column]])

        if val is not None:
            self.X_val = pd.DataFrame(val.loc[:, val.columns!=val.columns[y_column]])
            self.y_val = pd.Series(val.loc[:, val.columns[y_column]])

        self.y_column=y_column

        #self.X_test = pd.DataFrame(test.loc[:, test.columns!=test.columns[y_column]])
        #self.y_test = pd.Series(test.loc[:, train.columns[y_column]])

    def preprocess_train(self, X_var_selection, scale_Y=False):
        #tup = manager.create_mask(train, [0, 1, 2, 12], select=False)

        #Variable selection
        self.X_train = self.X_train.loc[:, X_var_selection]
        if(self.X_val is not None):
            self.X_val = self.X_val.loc[:, X_var_selection]
        #self.X_test = self.X_test.loc[:, X_var_selection]

        #Z-Score Normalization
        self.scaler_X = StandardScaler()
        self.scaler_X.fit(self.X_train)

        self.X_train = pd.DataFrame(self.scaler_X.transform(self.X_train))
        if (self.X_val is not None):
            self.X_val = pd.DataFrame(self.scaler_X.transform(self.X_val))

        if scale_Y:
            self.scaler_y = StandardScaler()
            self.scaler_y.fit(self.y_train.values.reshape(-1,1))

            self.y_train = pd.Series(self.scaler_y.transform(self.y_train.values.reshape(-1,1)).reshape(-1,))
            if (self.y_val is not None):
                self.y_val = pd.Series(self.scaler_y.transform(self.y_val.values.reshape(-1,1)).reshape(-1,))
            self.is_y_scaled=True
            return self.scaler_X, self.scaler_y

        self.is_y_scaled=False
        return self.scaler_X







    def preprocess_test(self,test,X_var_selection):
        X_test = test.loc[:, test.columns!=test.columns[self.y_column]]
        y_test = test.loc[:, test.columns[self.y_column]]

        #Variable selection
        X_test = X_test.loc[:, X_var_selection]

        #Z-Score Normalization
        X_test = pd.DataFrame(self.scaler_X.transform(X_test))
        if self.is_y_scaled:
            y_test = pd.Series(self.scaler_y.transform(y_test.values.reshape(-1,1)).reshape(-1,))

        return X_test, y_test


    def fit_linear_regression(self):
        #Due to is not necessary val set, we stack it to train set, in order to have more training instances
        if self.X_val is not None:
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train

        self.linear_reg_model = LinearRegression()
        self.linear_reg_model=self.linear_reg_model.fit(X_train_aux,y_train_aux)
        return self.linear_reg_model


    def predict_linear_regression(self,test_set,X_var_selection):
        X_test,y_test=self.preprocess_test(test_set,X_var_selection)
        if self.is_y_scaled:
            y_predicted=self.scaler_y.inverse_transform(self.linear_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.linear_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences=[r-p for r,p in zip(y_true,y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true":y_true,
            "y_predicted": y_predicted,
            "mae":mean_absolute_error(y_true, y_predicted),
            "mse":mean_squared_error(y_true,y_predicted),
            "r2":r2_score(y_true,y_predicted),
            "differences":differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }



    def fit_poly_regression(self, degree):
        #Due to is not necessary val set, we stack it to train set, in order to have more training instances
        if self.X_val is not None:
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train

        self.poly_X = PolynomialFeatures(degree=degree)
        X_train_aux = self.poly_X.fit_transform(X_train_aux)

        if self.is_y_scaled:
            self.poly_y = PolynomialFeatures(degree=degree)
            y_train_aux = self.poly_y.fit_transform(y_train_aux.values.reshape(-1,1))

        self.poly_reg_model = LinearRegression()
        self.poly_reg_model=self.poly_reg_model.fit(X_train_aux,y_train_aux)
        return self.poly_reg_model, self.poly_X, self.poly_y


    def predict_poly_regression(self,test_set,X_var_selection):
        X_test,y_test=self.preprocess_test(test_set,X_var_selection)
        X_test=self.poly_X.transform(X_test)

        if self.is_y_scaled:
            y_predicted=y_test = self.poly_y.inverse_transform(self.scaler_y.inverse_transform(self.poly_reg_model.predict(X_test)))
            y_true = self.poly_y.inverse_transform(self.scaler_y.inverse_transform(y_test))
        else:
            y_predicted = self.poly_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences=[r-p for r,p in zip(y_true,y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }


    def fit_dt_regression(self):#, max_depth):
        if(self.X_val is not None):
            X_train_aux=pd.concat([pd.DataFrame(self.X_train.copy()),pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(self.y_train.shape[0]+self.y_val.shape[0],))
        else:
            X_train_aux=self.X_train
            y_train_aux = self.y_train

        # self.dt_reg_model = DecisionTreeRegressor(max_depth=max_depth)
        # self.dt_reg_model=self.dt_reg_model.fit(X_train_aux,y_train_aux)

        dt = DecisionTreeRegressor()
        params = {
            "max_depth": [i for i in range(1, 50)]
        }
        self.gs_dt = RandomizedSearchCV(dt, params, n_jobs=-1, verbose=2)
        self.gs_dt.fit(X_train_aux, y_train_aux)

        self.dt_reg_model = self.gs_dt.best_estimator_

        return self.dt_reg_model

    def predict_dt_regression(self,test_set,X_var_selection):
        X_test,y_test=self.preprocess_test(test_set,X_var_selection)
        if self.is_y_scaled:
            y_predicted=self.scaler_y.inverse_transform(self.dt_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.dt_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences=[r-p for r,p in zip(y_true,y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "best_params": self.gs_dt.best_params_,
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }

    def fit_rf_regression(self):#, max_depth):
        if (self.X_val is not None):
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train

        rf = RandomForestRegressor(criterion="mse", max_features="sqrt")
        params = {
            "max_depth": [i for i in range(5, 55,5)],
            "n_estimators": [i*10 for i in range(5,55,5)]
        }
        self.gs_rf = RandomizedSearchCV(rf, params, n_jobs=-1, verbose=2)
        self.gs_rf.fit(X_train_aux, y_train_aux)

        self.rf_reg_model = self.gs_rf.best_estimator_#RandomForestRegressor(max_depth=max_depth, criterion="mse", max_features="sqrt")
        #self.rf_reg_model = self.rf_reg_model.fit(X_train_aux, y_train_aux)

        return self.rf_reg_model

    def predict_rf_regression(self,test_set,X_var_selection):
        X_test,y_test=self.preprocess_test(test_set,X_var_selection)
        if self.is_y_scaled:
            y_predicted=self.scaler_y.inverse_transform(self.rf_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.rf_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences=[r-p for r,p in zip(y_true,y_predicted)]
        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "best_params": self.gs_rf.best_params_,
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }

    def fit_boost_regression(self):#, max_depth, learning_rate=0.1):
        if (self.X_val is not None):
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train

        boost = GradientBoostingRegressor(loss='ls', max_features="sqrt",min_impurity_decrease=0.05*np.std(y_train_aux))
        params = {
            "max_depth": [i for i in range(5, 55, 5)],
            "learning_rate": [0.001,0.01,0.1],
            "n_estimators": [i * 10 for i in range(5, 55, 5)]
        }
        self.gs_boost = RandomizedSearchCV(boost, params, n_jobs=-1, verbose=2)
        self.gs_boost.fit(X_train_aux, y_train_aux)

        self.boost_reg_model = self.gs_boost.best_estimator_
        #self.boost_reg_model = self.boost_reg_model.fit(X_train_aux, y_train_aux)

        return self.boost_reg_model

    def predict_boost_regression(self,test_set,X_var_selection):
        X_test,y_test=self.preprocess_test(test_set,X_var_selection)
        if self.is_y_scaled:
            y_predicted=self.scaler_y.inverse_transform(self.boost_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.boost_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences=[r-p for r,p in zip(y_true,y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true":y_true,
            "y_predicted": y_predicted,
            "mae":mean_absolute_error(y_true, y_predicted),
            "mse":mean_squared_error(y_true,y_predicted),
            "r2":r2_score(y_true,y_predicted),
            "differences":differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "best_params": self.gs_boost.best_params_,
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }

    def fit_xgboost_regression(self):
        if (self.X_val is not None):
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train



        xgbreg = XGBRegressor(nthreads=-1)
        params = {
            "max_depth": [i for i in range(5,55,5)],
            "learning_rate": [0.001,0.01,0.1],
            "gamma": [i for i in range(1,20)],
            "n_estimators": [i * 10 for i in range(5, 55, 5)]
        }
        self.gs_xgboost = RandomizedSearchCV(xgbreg, params, n_jobs=-1,verbose=2)
        self.gs_xgboost.fit(X_train_aux, y_train_aux)

        self.xgboost_reg_model = self.gs_xgboost.best_estimator_
        #self.xgboost_reg_model = self.xgboost_reg_model.fit(X_train_aux, y_train_aux)

        return self.xgboost_reg_model

    def predict_xgboost_regression(self,test_set,X_var_selection):
        X_test, y_test = self.preprocess_test(test_set, X_var_selection)
        if self.is_y_scaled:
            y_predicted = self.scaler_y.inverse_transform(self.xgboost_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.xgboost_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences = [r - p for r, p in zip(y_true, y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "best_params": self.gs_xgboost.best_params_,
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }












    def fit_svm_regression(self):
        if (self.X_val is not None):
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train

        svmreg = SVR()

        params = {
            "C":[i*10 for i in range (1,11)],
            "epsilon": [1e-4,1e-3,1e-2,1e-1],
            #"degree":[i for i in range(3,5)], #3,4,5
            #"gamma": [i for i in range(3,5)], # 3,4,5,6
            "kernel": ["rbf","poly","sigmoid"],
        }
        self.gs_svm= RandomizedSearchCV(svmreg, params, n_jobs=-1,verbose=2)
        self.gs_svm.fit(X_train_aux, y_train_aux)

        self.svm_reg_model = self.gs_svm.best_estimator_
        #self.xgboost_reg_model = self.xgboost_reg_model.fit(X_train_aux, y_train_aux)

        #self.svm_reg_model = SVR(verbose=True)
        #self.svm_reg_model.fit(X_train_aux, y_train_aux)
        return self.svm_reg_model

    def predict_svm_regression(self, test_set, X_var_selection):
        X_test, y_test = self.preprocess_test(test_set, X_var_selection)
        if self.is_y_scaled:
            y_predicted = self.scaler_y.inverse_transform(self.svm_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.svm_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences = [r - p for r, p in zip(y_true, y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "best_params": self.gs_svm.best_params_,
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }





    def fit_mlp_regression(self):
        if (self.X_val is not None):
            X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
            y_train_aux = pd.Series(
                pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
                    self.y_train.shape[0] + self.y_val.shape[0], ))
        else:
            X_train_aux = self.X_train
            y_train_aux = self.y_train

        mlpreg = MLPRegressor(activation='relu')

        params = {
            "hidden_layer_sizes":[(200,),(250,),(300,),(350,),(400,)],
            "alpha": [0.0001,0.001,0.01],
            "learning_rate": ['adaptive'],
            "max_iter": [300],
        }

        self.gs_mlp= RandomizedSearchCV(mlpreg, params, n_jobs=-1,verbose=2)
        self.gs_mlp.fit(X_train_aux, y_train_aux)

        self.mlp_reg_model = self.gs_mlp.best_estimator_
        return self.mlp_reg_model



    def predict_mlp_regression(self,test_set,X_var_selection):
        X_test, y_test = self.preprocess_test(test_set, X_var_selection)
        if self.is_y_scaled:
            y_predicted = self.scaler_y.inverse_transform(self.mlp_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.mlp_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences = [r - p for r, p in zip(y_true, y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "best_params": self.gs_mlp.best_params_,
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }

    def fit_dnn_regression(self, topology, epochs, batch_size, save=False, save_dir=None, use_dropout=True, dropout_rate=0.2):
        """
            Possible AFs
            activation_functions = {
                    0: "relu",
                    1: "sigmoid",
                    2: "softmax",
                    3: "tanh",
                    4: "softplus",
                    5: "linear"
                }
            """
        from keras.layers import Dense, Dropout
        from keras.models import Sequential
        from keras.callbacks import ModelCheckpoint
        from keras.callbacks import EarlyStopping
        from keras.wrappers.scikit_learn import KerasRegressor
        from keras.optimizers import Adam

        def build_net():
            model = Sequential()
            model.add(Dense(topology[0][0], activation=topology[0][1], input_dim=self.X_train.shape[1], kernel_initializer='he_normal'))
            #if use_dropout:
            #    model.add(Dropout(dropout_rate))
            for i in range(1, len(topology)):
                model.add(Dense(topology[i][0], activation=topology[i][1], kernel_initializer='he_normal'))
                if use_dropout:
                    model.add(Dropout(dropout_rate))
            model.add(Dense(1, kernel_initializer='he_normal'))

            model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=True), loss='mse') #Non negative Smu
            return model

        save_ok=False
        if save:
            if save_dir is not None:
                if not os.path.isdir(save_dir):
                    raise NotADirectoryError('{} is not a directory.'.format(save_dir))
                save_ok=True

        self.dnn_reg_model = KerasRegressor(build_fn=build_net, epochs=epochs, batch_size=batch_size, verbose=1)
        #early_stop = EarlyStopping("val_loss", min_delta=1e-2, patience=50, verbose=1, mode="min")
        #callback_list = [early_stop]
        callback_list =[]

        if save_ok:
            checkpoint = ModelCheckpoint(save_dir + 'weights.h5', 'val_loss', verbose=1, mode="min",save_best_only=True)
            callback_list.append(checkpoint)

        if self.X_val is None:
            self.dnn_reg_model.fit(self.X_train, self.y_train, validation_split=0.2,callbacks=callback_list)
        else:
            self.dnn_reg_model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),callbacks=callback_list)

        if save_ok:
            self.dnn_reg_model.model.save(save_dir+'net.h5')

        return self.dnn_reg_model


    def predict_dnn_regression(self,test_set,X_var_selection):
        X_test, y_test = self.preprocess_test(test_set, X_var_selection)
        if self.is_y_scaled:
            y_predicted = self.scaler_y.inverse_transform(self.dnn_reg_model.predict(X_test))
            y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
        else:
            y_predicted = self.dnn_reg_model.predict(X_test)
            y_true = y_test.values.reshape(-1, )

        differences = [r - p for r, p in zip(y_true, y_predicted)]

        fold_spearman_rho, fold_spearman_pval = scipy.stats.spearmanr(y_true, y_predicted)
        fold_pearson_rho, fold_pearson_pval = scipy.stats.pearsonr(y_true, y_predicted)

        return {
            "y_true": y_true,
            "y_predicted": y_predicted,
            "mae": mean_absolute_error(y_true, y_predicted),
            "mse": mean_squared_error(y_true, y_predicted),
            "r2": r2_score(y_true, y_predicted),
            "differences": differences,
            "difference_mean": np.mean([differences]),
            "difference_std": np.std([differences]),
            "spearman_rho": fold_spearman_rho,
            "spearman_pval": fold_spearman_pval,
            "pearson_rho": fold_pearson_rho,
            "pearson_pval": fold_pearson_pval,
        }




    # def fit_gp_regression(self,kernel=None):
    #
    #     if (self.X_val is not None):
    #         X_train_aux = pd.concat([pd.DataFrame(self.X_train.copy()), pd.DataFrame(self.X_val.copy())])
    #         y_train_aux = pd.Series(
    #             pd.concat([pd.DataFrame(self.y_train.copy()), pd.DataFrame(self.y_val.copy())]).values.reshape(
    #                 self.y_train.shape[0] + self.y_val.shape[0], ))
    #     else:
    #         X_train_aux = self.X_train
    #         y_train_aux = self.y_train
    #
    #     if kernel == "linear":
    #         kernel = GPy.kern.Linear(X_train_aux.shape[1])  # , ARD=True)
    #     else:
    #         kernel = None
    #         # kernel=None #RBF by default
    #     self.gproc_reg_model = GPy.models.GPRegression(X_train_aux, y_train_aux.values.reshape(-1,1), kernel=kernel)  # Kernel RBF by default
    #     self.gproc_reg_model.optimize(verbose=True)
    #
    #     return self.gproc_reg_model
    #
    # def predict_gp_regression(self,test_set,X_var_selection):
    #     X_test, y_test = self.preprocess_test(test_set, X_var_selection)
    #     if self.is_y_scaled:
    #         y_predicted = self.scaler_y.inverse_transform(self.gproc_reg_model.predict(X_test))
    #         y_true = self.scaler_y.inverse_transform(y_test.values.reshape(-1, ))
    #     else:
    #         y_predicted = self.gproc_reg_model.predict(X_test)
    #         y_true = y_test.values.reshape(-1, )
    #
    #     differences = [r - p for r, p in zip(y_true, y_predicted)]
    #
    #     return {
    #         "y_true": y_true,
    #         "y_predicted": y_predicted,
    #         "mae": mean_absolute_error(y_true, y_predicted),
    #         "mse": mean_squared_error(y_true, y_predicted),
    #         "r2": r2_score(y_true, y_predicted),
    #         "differences": differences,
    #         "difference_mean": np.mean([differences]),
    #         "difference_std": np.std([differences]),
    #         "best_params": self.gs_gproc.best_params_,
    #     }
