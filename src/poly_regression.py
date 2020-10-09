import global_dirs
import pandas as pd
import numpy as np

from model_manager import ModelManager

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os

from sklearn.externals import joblib

manager=ModelManager()
train=pd.concat(manager.read_data(global_dirs.splitted_data_path, formats=["hdf"], type="train",verbose=False)[1].values())
validation=pd.concat(manager.read_data(global_dirs.splitted_data_path, formats=["hdf"], type="validation",verbose=False)[1].values())

manager.assign_sets(train=train)
tup = manager.create_mask(train.iloc[:,:-1], global_dirs.variable_selection[0], select=global_dirs.variable_selection[1]) #This tuple shouldn't take care about y_column index
scalers=manager.preprocess_train(tup,scale_Y=False)


degree_tuning=np.arange(1,7) # We test until degree 6
best_degree=1
best_mse=None
history_mse=[]
for deg in degree_tuning:
   manager.fit_poly_regression(degree=deg)
   mse=manager.predict_poly_regression(validation, tup)["mse"]
   if best_mse is None or (abs(mse) < abs(best_mse)): #and r2>=-1 and r2<=1) or best_r2 is None:
       best_degree=deg
       best_mse=mse
   history_mse.append(mse)
   print("{}: {}".format(deg,mse))

poly_model, poly_transform_x, poly_transform_y = manager.fit_poly_regression(degree=best_degree)

if not os.path.isdir(global_dirs.results_path):
    os.mkdir(global_dirs.results_path)
if not os.path.isdir(global_dirs.poly_regression_path):
    os.mkdir(global_dirs.poly_regression_path)
if not os.path.isdir(global_dirs.poly_regression_path+"scalers/"):
    os.mkdir(global_dirs.poly_regression_path+"scalers/")
if not os.path.isdir(global_dirs.poly_regression_path+"model/"):
    os.mkdir(global_dirs.poly_regression_path+"model/")
if not os.path.isdir(global_dirs.poly_regression_path + "results/"):
    os.mkdir(global_dirs.poly_regression_path + "results/")

if isinstance(scalers,tuple):
    joblib.dump(scalers[0], global_dirs.poly_regression_path + "scalers/scaler_X.h5")
    joblib.dump(scalers[1], global_dirs.poly_regression_path + "scalers/scaler_y.h5")
    joblib.dump(poly_transform_x, global_dirs.poly_regression_path + "scalers/poly_transform_x.h5")
    if poly_transform_y is not None:
        joblib.dump(poly_transform_y, global_dirs.poly_regression_path + "scalers/poly_transform_y.h5")
else:
    joblib.dump(scalers, global_dirs.poly_regression_path + "scalers/scaler_X.h5")
joblib.dump(poly_model, global_dirs.poly_regression_path + "/model/model.h5")


test={name.split("-")[1]: data for name,data in manager.read_data(global_dirs.splitted_data_path, formats=["hdf"], sim="qgsjet",train="test",verbose=False)[1].items()}

colors={
    "proton":"magenta",
    "helium":"red",
    "oxygen":"green",
    "iron":"orange",}

dec_round=3

plt.rc('axes', labelsize=25)
plt.rc('axes', titlesize=25)
plt.rc('legend', fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)


plt.plot(degree_tuning,history_mse,"o-g",lw=2)
plt.title("MSE for degree parameter in validation")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.axvline(x=best_degree)
best_depth_patch = mpatches.Patch(color='white', label='Best degree: {}'.format(best_degree))
best_r2_patch = mpatches.Patch(color='white', label='Best mse: {}'.format(round(best_mse,dec_round)))
plt.legend(handles=[best_depth_patch,best_r2_patch],loc="lower right")
plt.savefig(global_dirs.poly_regression_path + "results/max_degree.png",dpi=70)
#plt.show()
plt.close()


results=None

for name,ds in test.items():
    results=manager.predict_poly_regression(ds,tup)

    f, ax = plt.subplots(1, 2)
    # Histogram of differences
    ax[0].hist(results["differences"], color=colors[name], histtype="step", lw=2)
    ax[0].axvline(x=0)
    ax[0].set_title("Poly Regression\nQGSJET-II (test)\nHistogram of differences - {}".format(name))
    ax[0].set_xlabel(r'$S_{\mu}^{real} - S_{\mu}^{pred}$')
    ax[0].set_ylabel("Count")
    dmean_patch = mpatches.Patch(color='white',
                                 label='Diff. mean: {}'.format(round(results["difference_mean"], dec_round)))
    dstd_patch = mpatches.Patch(color='white',
                                label='Diff. std: {}'.format(round(results["difference_std"], dec_round)))
    ax[0].legend(handles=[dmean_patch, dstd_patch], loc="upper left")
    ax[0].grid(alpha=0.5)

    # fit = np.polyfit(results["y_true"], results["y_predicted"], 1)
    # fit_fn = np.poly1d(fit)
    ax[1].scatter(results["y_true"], results["y_predicted"], c=colors[name], marker="o")
    # ax[1].plot(results["y_true"], fit_fn(results["y_true"]), '--b', lw=2)
    top_axis = int(max(max(results["y_true"]), max(results["y_predicted"]))) + 1
    identity_coords = [i for i in range(top_axis)]
    ax[1].plot(identity_coords, identity_coords, '--b', lw=2)
    ax[1].set_title("Poly Regression\nQGSJET-II (test)\nReal vs. Predicted - {}".format(name))
    ax[1].set_xlabel(r'$S_{\mu}^{real}$')
    ax[1].set_ylabel(r'$S_{\mu}^{predicted}$')
    r2_patch = mpatches.Patch(color='white', label='R2: {}'.format(round(results["r2"], dec_round)))
    mae_patch = mpatches.Patch(color='white', label='MAE: {}'.format(round(results["mae"], dec_round)))
    mse_patch = mpatches.Patch(color='white', label='MSE: {}'.format(round(results["mse"], dec_round)))
    rho_patch = mpatches.Patch(color='white',
                               label=r"Pearson's $\rho$: {}".format(round(results["pearson_rho"], dec_round)))
    pval_patch = mpatches.Patch(color='white',
                                label=r"Pearson's $\rho$ p-val: {}".format(round(results["pearson_pval"], dec_round)))
    ax[1].legend(handles=[r2_patch, mae_patch, mse_patch, rho_patch, pval_patch], loc="lower right")
    ax[1].grid(alpha=0.5)
    # plt.show()
    plt.savefig(global_dirs.poly_regression_path + "results/qgsjet-results-{}.png".format(name), dpi=70)


test={name.split("-")[1]: data for name,data in manager.read_data(global_dirs.splitted_data_path, formats=["hdf"], sim="epos", train="test",verbose=False)[1].items()}

for name,ds in test.items():

    results=manager.predict_poly_regression(ds,tup)


    f, ax = plt.subplots(1, 2)
    #Histogram of differences
    ax[0].hist(results["differences"], color=colors[name],histtype="step",lw=2)
    ax[0].axvline(x=0)
    ax[0].set_title("Poly Regression\nEPOS (test)\nHistogram of differences - {}".format(name))
    ax[0].set_xlabel(r'$S_{\mu}^{real} - S_{\mu}^{pred}$')
    ax[0].set_ylabel("Count")
    dmean_patch = mpatches.Patch(color='white', label='Diff. mean: {}'.format(round(results["difference_mean"],dec_round)))
    dstd_patch = mpatches.Patch(color='white', label='Diff. std: {}'.format(round(results["difference_std"],dec_round)))
    ax[0].legend(handles=[dmean_patch, dstd_patch], loc="upper left")
    ax[0].grid(alpha=0.5)

    # fit = np.polyfit(results["y_true"], results["y_predicted"], 1)
    # fit_fn = np.poly1d(fit)
    ax[1].scatter(results["y_true"], results["y_predicted"], c=colors[name], marker="o")
    # ax[1].plot(results["y_true"], fit_fn(results["y_true"]), '--b', lw=2)
    top_axis = int(max(max(results["y_true"]), max(results["y_predicted"]))) + 1
    identity_coords = [i for i in range(top_axis)]
    ax[1].plot(identity_coords, identity_coords, '--b', lw=2)
    ax[1].set_title("Poly Regression\nEPOS (test)\nReal vs. Predicted - {}".format(name))
    ax[1].set_xlabel(r'$S_{\mu}^{real}$')
    ax[1].set_ylabel(r'$S_{\mu}^{predicted}$')
    r2_patch = mpatches.Patch(color='white', label='R2: {}'.format(round(results["r2"],dec_round)))
    mae_patch = mpatches.Patch(color='white', label='MAE: {}'.format(round(results["mae"],dec_round)))
    mse_patch = mpatches.Patch(color='white', label='MSE: {}'.format(round(results["mse"],dec_round)))
    rho_patch = mpatches.Patch(color='white',
                               label=r"Pearson's $\rho$: {}".format(round(results["pearson_rho"], dec_round)))
    pval_patch = mpatches.Patch(color='white',
                                label=r"Pearson's $\rho$ p-val: {}".format(round(results["pearson_pval"], dec_round)))
    ax[1].legend(handles=[r2_patch, mae_patch, mse_patch, rho_patch, pval_patch], loc="lower right")
    ax[1].grid(alpha=0.5)


    #plt.show()
    plt.savefig(global_dirs.poly_regression_path + "results/epos-results-{}.png".format(name), dpi=70)

with open(global_dirs.poly_regression_path+"best_params.txt","w") as f:
    f.write("Best params: {}\n".format({'degree':best_degree}))

import pickle
best_params={"degree":best_degree}
f = open(global_dirs.poly_regression_path+'best_params.pkl', 'wb')
pickle.dump(best_params, f)
f.close()











#TEST COMPLETO
test={name.split("-")[1]: data for name,data in manager.read_data(global_dirs.splitted_data_path_real, formats=["hdf"], sim="qgsjet",train="test",verbose=False)[1].items()}

results=None

for name,ds in test.items():
    results=manager.predict_poly_regression(ds,tup)

    f, ax = plt.subplots(1, 2)
    # Histogram of differences
    ax[0].hist(results["differences"], color=colors[name], histtype="step", lw=2)
    ax[0].axvline(x=0)
    ax[0].set_title("Poly Regression\nQGSJET-II (test)\nHistogram of differences - {}".format(name))
    ax[0].set_xlabel(r'$S_{\mu}^{real} - S_{\mu}^{pred}$')
    ax[0].set_ylabel("Count")
    dmean_patch = mpatches.Patch(color='white',
                                 label='Diff. mean: {}'.format(round(results["difference_mean"], dec_round)))
    dstd_patch = mpatches.Patch(color='white',
                                label='Diff. std: {}'.format(round(results["difference_std"], dec_round)))
    ax[0].legend(handles=[dmean_patch, dstd_patch], loc="upper left")
    ax[0].grid(alpha=0.5)

    # fit = np.polyfit(results["y_true"], results["y_predicted"], 1)
    # fit_fn = np.poly1d(fit)
    ax[1].scatter(results["y_true"], results["y_predicted"], c=colors[name], marker="o")
    # ax[1].plot(results["y_true"], fit_fn(results["y_true"]), '--b', lw=2)
    top_axis = int(max(max(results["y_true"]), max(results["y_predicted"]))) + 1
    identity_coords = [i for i in range(top_axis)]
    ax[1].plot(identity_coords, identity_coords, '--b', lw=2)
    ax[1].set_title("Poly Regression\nQGSJET-II (test)\nReal vs. Predicted - {}".format(name))
    ax[1].set_xlabel(r'$S_{\mu}^{real}$')
    ax[1].set_ylabel(r'$S_{\mu}^{predicted}$')
    r2_patch = mpatches.Patch(color='white', label='R2: {}'.format(round(results["r2"], dec_round)))
    mae_patch = mpatches.Patch(color='white', label='MAE: {}'.format(round(results["mae"], dec_round)))
    mse_patch = mpatches.Patch(color='white', label='MSE: {}'.format(round(results["mse"], dec_round)))
    rho_patch = mpatches.Patch(color='white',
                               label=r"Pearson's $\rho$: {}".format(round(results["pearson_rho"], dec_round)))
    pval_patch = mpatches.Patch(color='white',
                                label=r"Pearson's $\rho$ p-val: {}".format(round(results["pearson_pval"], dec_round)))
    ax[1].legend(handles=[r2_patch, mae_patch, mse_patch, rho_patch, pval_patch], loc="lower right")
    ax[1].grid(alpha=0.5)
    # plt.show()
    plt.savefig(global_dirs.poly_regression_path + "results/qgsjet-results-WHOLE-{}.png".format(name), dpi=70)
