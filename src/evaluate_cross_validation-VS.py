import pandas as pd
import global_dirs
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_dataframe(df,title,mask,save_dir,alpha=0.05,cmap="Blues"):
    sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f",mask=mask,vmin=0, vmax=1.2*alpha, cbar=False)
    plt.title(title)
    plt.savefig(save_dir, dpi=100)
    #plt.show()
    plt.close()

def compute_results(results,name):
    res=results.iloc[:,1:]
    mean_results=res.mean().to_dict()
    print(name)
    print("\tR2:{}".format(mean_results["r2"]))
    print("\tRHO Pearson: {}".format(mean_results["pearson_rho"]))


def anova_test(results_1,name_1,results_2,name_2, alpha=0.05,verbose=False):
    res1 = results_1.iloc[:, 1:]
    res2 = results_2.iloc[:, 1:]

    if verbose:print("ANOVA TEST FOR {} vs. {}".format(name_1,name_2))
    measures = {"mae": 0, "mse": 0, "pearson_rho": 0, "r2": 0}
    for col in measures.keys():
        statistic,pvalue=scipy.stats.f_oneway(res1[col], res2[col])
        if pvalue <= alpha:
            if verbose:print("\t{} and {} ARE STATISTICALLY DIFFERENT in {} measure. P-value: {}".format(name_1,name_2,col,pvalue))
        else:
            if verbose:print("\t{} and {} ARE NOT STATISTICALLY DIFFERENT in {} measure. P-value: {}".format(name_1, name_2,col, pvalue))
        measures[col] = pvalue

    if verbose:print("\t---------Cross validation {} mean results---------".format(name_1))
    #print(res1)
    if verbose:print(res1.mean()[measures])
    if verbose:print(res1.std()[measures])
    if verbose:print("\t---------Cross validation {} mean results---------".format(name_2))
    #print(res2)
    if verbose:print(res2.mean()[measures])
    if verbose:print(res2.std()[measures])
    return measures


def kruskal_test(results_1,name_1,results_2,name_2, alpha=0.05, verbose=False):
    res1 = results_1.iloc[:, 1:]
    res2 = results_2.iloc[:, 1:]

    if verbose:print("KRUSKAL TEST FOR {} vs. {}".format(name_1,name_2))
    measures = {"mae":0,"mse":0,"pearson_rho":0,"r2":0}
    for col in measures.keys():
        statistic,pvalue=scipy.stats.kruskal(res1[col], res2[col])
        if pvalue <= alpha:
            if verbose:print("\t{} and {} ARE STATISTICALLY DIFFERENT in {} measure. P-value: {}".format(name_1,name_2,col,pvalue))
        else:
            if verbose:print("\t{} and {} ARE NOT STATISTICALLY DIFFERENT in {} measure. P-value: {}".format(name_1, name_2,col, pvalue))
        measures[col]=pvalue

    if verbose:print("\t---------Cross validation {} mean results---------".format(name_1))
    #print(res1)
    if verbose:print(res1.mean()[measures])
    if verbose:print(res1.std()[measures])
    if verbose:print("\t---------Cross validation {} mean results---------".format(name_2))
    #print(res2)
    if verbose:print(res2.mean()[measures])
    if verbose:print(res2.std()[measures])
    return measures


def shapiro_test(results,name):
    measures = {"mae": 0, "mse": 0, "pearson_rho": 0, "r2": 0}
    print("Shapiro normality test for {}:".format(name))
    for criteria in measures.keys():
        measures[criteria]=scipy.stats.shapiro(results.loc[:criteria])
        print("\t{} P-value: {}".format(criteria,measures[criteria]))
    return measures

def plot_summary(results):
    for (res,name) in results:
        print(res.iloc[:,1:].mean(),name)


cv_linear_reg=pd.read_csv(global_dirs.linear_regression_path+"results/100-fold-cross-validation-results.csv")
cv_poly_reg=pd.read_csv(global_dirs.poly_regression_path+"results/100-fold-cross-validation-results.csv")
cv_dt=pd.read_csv(global_dirs.decision_tree_path+"results/100-fold-cross-validation-results.csv")
cv_rf=pd.read_csv(global_dirs.random_forest_path+"results/100-fold-cross-validation-results.csv")
#cv_gb=pd.read_csv(global_dirs.gradient_boosting_path+"results/100-fold-cross-validation-results.csv")
cv_xgb=pd.read_csv(global_dirs.xgboost_path+"results/100-fold-cross-validation-results.csv")
cv_svm=pd.read_csv(global_dirs.svm_path+"results/100-fold-cross-validation-results.csv")
cv_mlp=pd.read_csv(global_dirs.mlp_path+"results/100-fold-cross-validation-results.csv")
#cv_dnn=pd.read_csv(global_dirs.dnn_path+"results/100-fold-cross-validation-results.csv")

cv_linear_reg_VS2=pd.read_csv(global_dirs.linear_regression_path_VS2+"results/100-fold-cross-validation-results.csv")
cv_poly_reg_VS2=pd.read_csv(global_dirs.poly_regression_path_VS2+"results/100-fold-cross-validation-results.csv")
cv_dt_VS2=pd.read_csv(global_dirs.decision_tree_path_VS2+"results/100-fold-cross-validation-results.csv")
cv_rf_VS2=pd.read_csv(global_dirs.random_forest_path_VS2+"results/100-fold-cross-validation-results.csv")
#cv_gb=pd.read_csv(global_dirs.gradient_boosting_path+"results/100-fold-cross-validation-results.csv")
cv_xgb_VS2=pd.read_csv(global_dirs.xgboost_path_VS2+"results/100-fold-cross-validation-results.csv")
cv_svm_VS2=pd.read_csv(global_dirs.svm_path_VS2+"results/100-fold-cross-validation-results.csv")
cv_mlp_VS2=pd.read_csv(global_dirs.mlp_path_VS2+"results/100-fold-cross-validation-results.csv")
#




results=[(cv_linear_reg,"Linear regression"),
#         (cv_poly_reg,"Poly regression"),
         (cv_dt,"Decision tree"),
         (cv_rf,"Random forest"),
#         (cv_gb,"Gradient Boosting"),
         (cv_xgb,"XGBoost"),
         (cv_svm,"SVR"),
         (cv_mlp,"MLP"),
         (cv_linear_reg_VS2,"Linear regression VS2"),
#         (cv_poly_reg_VS2,"Poly regression VS2"),
         (cv_dt_VS2,"Decision tree VS2"),
         (cv_rf_VS2,"Random forest VS2"),
#         (cv_gb,"Gradient Boosting"),
         (cv_xgb_VS2,"XGBoost VS2"),
         (cv_svm_VS2,"SVR VS2"),
         (cv_mlp_VS2,"MLP VS2"),
#         (cv_dnn,"DNN")
]



#plot_summary(results)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc('figure',figsize=(35,29))
plt.rc('figure',autolayout=True)
plt.rc('axes', labelsize=35)
plt.rc('axes', titlesize=35)
plt.rc('legend', fontsize=32)
plt.rc('xtick', labelsize=35,direction='inout')
plt.rc('ytick', labelsize=35)
plt.rc('font', size=35)



heatmap_data=pd.DataFrame.from_dict({name:r.iloc[:,1:].mean() for r,name in results}).loc[["mae","mse","r2","pearson_rho"],[name for r,name in results]]
heatmap_center=np.mean(heatmap_data.mean().values)


#ax = plt.subplot()
#ax=sns.heatmap(heatmap_data,cmap="coolwarm",annot=True,fmt=".2f",cbar=False,robust=True,linewidths=.5,center=heatmap_center)
#ax.set_xticklabels([name for r,name in results],rotation=60)
#ax.set_yticklabels(["MAE\n(lower better)","MSE\n(lower better)",r"$R^2$"+"\n(higher better)",r"Pearson's $\rho$"+"\n(higher better)"],rotation=0)
#plt.show()




pd.set_option('display.width', 500)



for r,n in results:
    measures=shapiro_test(r,n)





interest_fields=['mae','mse','pearson_rho','pearson_pval','r2']
for r,n in results:
    print("--- {} ---".format(n))
    print(r.iloc[:,1:].round(2).loc[:,interest_fields])
    print(r.mean().round(2).loc[interest_fields])
    print(r.std().round(2).loc[interest_fields])



algorithm_names=[r[1] for r in results]



mat_kruskal_mae=np.zeros((len(results),len(results)))
mat_kruskal_mse=np.zeros((len(results),len(results)))
mat_kruskal_pearson_rho=np.zeros((len(results),len(results)))
mat_kruskal_r2=np.zeros((len(results),len(results)))
mat_anova_mae=np.zeros((len(results),len(results)))
mat_anova_mse=np.zeros((len(results),len(results)))
mat_anova_pearson_rho=np.zeros((len(results),len(results)))
mat_anova_r2=np.zeros((len(results),len(results)))

for i,r1 in enumerate(results):
    for j,r2 in enumerate(results):
        if(i>j):
            measures=kruskal_test(r1[0],r1[1],r2[0],r2[1],verbose=True)
            mat_kruskal_mae[i,j]=measures["mae"]
            mat_kruskal_mse[i,j]=measures["mse"]
            mat_kruskal_pearson_rho[i,j]=measures["pearson_rho"]
            mat_kruskal_r2[i,j]=measures["r2"]
            print("#################################################")
            measures = anova_test(r1[0], r1[1], r2[0], r2[1],verbose=True)
            mat_anova_mae[i, j] = measures["mae"]
            mat_anova_mse[i, j] = measures["mse"]
            mat_anova_pearson_rho[i, j] = measures["pearson_rho"]
            mat_anova_r2[i, j] = measures["r2"]
            print("#################################################")





"""plt.rc('axes', labelsize=25)
plt.rc('axes', titlesize=25)
plt.rc('legend', fontsize=15)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('font', size=15)
"""
mat_kruskal_mae=pd.DataFrame(mat_kruskal_mae, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_kruskal_mse=pd.DataFrame(mat_kruskal_mse, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_kruskal_pearson_rho=pd.DataFrame(mat_kruskal_pearson_rho, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_kruskal_r2=pd.DataFrame(mat_kruskal_r2, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_anova_mae=pd.DataFrame(mat_anova_mae, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_anova_mse=pd.DataFrame(mat_anova_mse, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_anova_pearson_rho=pd.DataFrame(mat_anova_pearson_rho, columns=[r[1] for r in results],index=[r[1] for r in results])
mat_anova_r2=pd.DataFrame(mat_anova_r2, columns=[r[1] for r in results],index=[r[1] for r in results])


mask = np.zeros_like(mat_kruskal_r2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask,True)




plot_dataframe(mat_anova_r2,"R2 p-value matrix for ANOVA test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-r2-anova.png",alpha=0.05)
plot_dataframe(mat_anova_mse,"MSE p-value matrix for ANOVA test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-mse-anova.png",alpha=0.05)
plot_dataframe(mat_anova_mae,"MAE p-value matrix for ANOVA test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-mae-anova.png",alpha=0.05)
plot_dataframe(mat_anova_pearson_rho,r"Pearson's $\rho$ p-value matrix for ANOVA test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-pearson_rho-anova.png",alpha=0.05)

plot_dataframe(mat_kruskal_r2,"R2 p-value matrix for Kruskal test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-r2-kruskal.png",alpha=0.05)
plot_dataframe(mat_kruskal_mse,"MSE p-value matrix for Kruskal test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-mse-kruskal.png",alpha=0.05)
plot_dataframe(mat_kruskal_mae,"MAE p-value matrix for Kruskal test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-mae-kruskal.png",alpha=0.05)
plot_dataframe(mat_kruskal_pearson_rho,r"Pearson's $\rho$ p-value matrix for Kruskal test",cmap="Blues_r",mask=mask,save_dir=global_dirs.results_path+"cv-pearson_rho-kruskal.png",alpha=0.05)


