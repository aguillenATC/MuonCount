from data_reader import DataReader
import global_dirs
import pandas as pd
import os

reader=DataReader()


_,data=reader.read_data(global_dirs.data_path,
                 filename_separator="-",
                 header=None,
                 formats=["txt"],
                 type="test",
                 simulator="QGSJet")

if not os.path.exists(global_dirs.splitted_data_path_real):
    os.makedirs(global_dirs.splitted_data_path_real)

for filename,ds in data.items():
    ds.to_hdf(global_dirs.splitted_data_path_real + "qgsjet-"+filename.split("-")[1] + "-test.hdf", "train", mode='w')