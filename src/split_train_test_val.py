from data_reader import DataReader
import global_dirs
import pandas as pd
import os


def split_train_test_validation(ds, name, dir_to_save, y_column=None, group_by_events=False, val_size=0.1, test_size=0.2):
    if y_column is None:
        y_column=-1

    if group_by_events:
        ds = ds.groupby(["event_id", "sim_id"]).mean().reset_index()



    X_training, y_training, X_val, y_val = reader.split_train_test(ds, test_size=val_size, y_column=y_column)
    validation = pd.concat([X_val, y_val], axis=1)
    rest_of_data = pd.concat([X_training, y_training], axis=1)

    test_size /= (1 - val_size)

    X_training, y_training, X_test, y_test = reader.split_train_test(rest_of_data, test_size=test_size, y_column=y_column)
    train= pd.concat([X_training, y_training], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    # Dataset are NOT stored filtered
    validation.to_hdf(dir_to_save + name + "-validation.hdf", "validation", mode='w')
    train.to_hdf(dir_to_save + name + "-train.hdf", "train", mode='w')
    test.to_hdf(dir_to_save + name + "-test.hdf", "test", mode='w')

    val_p = validation.shape[0] / ds.shape[0]
    tr_p = train.shape[0] / ds.shape[0]
    ts_p = test.shape[0] / ds.shape[0]


    print("Proporcion conseguida en validacion: {}".format(round(val_p,2)))
    print("Proporcion conseguida en test: {}".format(round(ts_p,2)))
    print("Proporcion conseguida en training: {}".format(round(tr_p,2)))
    print("Total: {}".format(val_p + tr_p + ts_p))
    print("Variables:\n{}".format(train.columns.values))

    #return train, validation, test



reader = DataReader()

_,data=reader.read_data(global_dirs.data_path,
                 filename_separator="-",
                 header=None,
                 formats=["txt"],
                 type="train",
                 simulator="QGSJet")

#data=pd.concat(data.values())

for filename,ds in data.items():
    split_train_test_validation(ds, "qgsjet-"+filename.split("-")[1], global_dirs.splitted_data_path, y_column=-1,group_by_events=False)


_,data=reader.read_data(global_dirs.data_path,
                    filename_separator="-",
                header=None,
               formats=["txt"],
               type="test",
               simulator="EPOS")

for filename,ds in data.items():
   ds.to_hdf(global_dirs.splitted_data_path + "epos-"+filename.split("-")[1]+ "-test.hdf", "epos_test", mode='w')
