from data_reader import DataReader
import global_dirs
import pandas as pd
import os

import numpy as np
from sklearn.model_selection import KFold # import KFold

def split_cross_validation(ds, name, dir_to_save, nfolds=5, group_by_events=False):
    if group_by_events:
        ds = ds.groupby(["event_id", "sim_id"]).mean().reset_index()

    kf = KFold(n_splits=nfolds)  # Define the split - into 2 folds


    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    # Datasets are NOT stored filtered
    for i, tup in enumerate(kf.split(ds)):
        train_idx = tup[0]
        test_idx = tup[1]
        ds.iloc[test_idx, :].to_hdf(dir_to_save + name + "-test-fold-{}.hdf".format(i),"test_fold_{}".format(i), mode="w")
        ds.iloc[train_idx, :].to_hdf(dir_to_save + name + "-train-fold-{}.hdf".format(i), "train_fold_{}".format(i),mode="w")


reader = DataReader()


_,data=reader.read_data(global_dirs.data_path,
                 filename_separator="-",
                 header=None,
                 formats=["txt"],
                 type="train",
                 simulator="QGSJet")

data=pd.concat(data.values())

split_cross_validation(data, "qsgjet", global_dirs.cross_val_data_path, nfolds=5)

_,data=reader.read_data(global_dirs.data_path,
                 filename_separator="-",
                 header=None,
                 formats=["txt"],
                 type="test",
                 simulator="EPOS")

for filename,ds in data.items():
    ds.to_hdf(global_dirs.cross_val_data_path + "epos-"+filename.split("-")[1]+ "-test.hdf", "epos_test", mode='w')