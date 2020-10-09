import global_dirs
from data_reader import DataReader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


reader=DataReader()
filenames,train=reader.read_data(global_dirs.splitted_data_path, formats=["hdf"], type="train")


#scaler=StandardScaler()
#train=pd.DataFrame(scaler.fit_transform(pd.concat(train.values())))


train=pd.concat(train.values())

train.columns=["EventID","SimID","Run Number","MCEnergy","MCZenith","Distance core","Total S","Trace length",
               "Azimuth","Risetime","Falltime","Area Over Peak","True Smu"]

#train.loc[:,reader.create_mask(train,[0,1,2,7,8,10],select=False)].plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#plt.show()
#plt.close()


energy=train.iloc[:1000,3]
smu=train.iloc[:1000,-1]
energy_not_log=np.power(10,energy)



print(np.min(energy.values))
print(np.max(energy.values))