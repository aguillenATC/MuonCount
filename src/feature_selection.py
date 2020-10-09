import global_dirs
import pandas as pd
from data_reader import DataReader
import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix
import seaborn as sns
from xgboost import plot_importance
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
#import pymrmr



reader = DataReader()
scaler=StandardScaler()


_,data=reader.read_data(global_dirs.data_path,
                 filename_separator="-",
                 header=None,
                 formats=["txt"],
                 type="train")
train=pd.concat(data.values())
#train.columns=["EventID","SimID","Run Number","MCEnergy","MCZenith","Distance core","Total S","Trace length",
#               "Azimuth","Risetime","Falltime","Area Over Peak","True Smu"]

# Remove the first three columns related to sim. run information
train = train.iloc[:,3:]

print(train.shape)

train.columns=["MCEnergy","MCZenith","Distance core","Total S","Trace length",
               "Azimuth","Risetime","Falltime","Area Over Peak","True Smu"]

X_train=pd.DataFrame(scaler.fit_transform(train.loc[:,train.columns!=train.columns[-1]]),columns = train.columns[:-1] )
y=train.loc[:,train.columns[-1]]



plt.rc('axes', labelsize=35)
plt.rc('axes', titlesize=35)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('font', size=15)


#model = XGBRegressor()
#model.fit(X,y)

#Dibujar importancia
#plot_importance(model)
#plt.show()
#plt.close()


#print(pymrmr.mRMR(X,'MIQ',12))


#Dibujar distribuciones de probabilidad

#train.loc[:,reader.create_mask(train,[0,1,2,7,8,10],select=False)].plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#plt.show()
#plt.close()

#Dibujar matriz de dispersion
#scatter_matrix(train)
#plt.show()

#Dibujar matriz de correlacion
import numpy as np
mask = np.zeros((train.shape[1],train.shape[1]), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask,True)
sns.heatmap(train.corr(),cmap="Blues",annot=True,fmt=".2f",cbar=False,mask=mask)
plt.show()
plt.close()