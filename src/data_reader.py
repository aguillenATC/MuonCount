import pandas as pd
import numpy as np
import os


class DataReader:
    def __init__(self):
        pass

    @classmethod
    def directory_to_hdf(self, data_path, sep_regex):
        """
        This function allows you to convert each file into a directory containing only .dat files to hdf.
        USE ONLY IF SEPARATOR ARE THE SAME FOR ALL DAT FILES.
        :param data_path: where the dat files are located.
        :param sep_regex: regular expression that describe row elements separator in an example (row).
        :return:
        """
        if (data_path[-1] != "/"):
            data_path = data_path + "/"

        files = os.listdir(data_path)
        for filename in files:
            extension=filename.split(".")[-1]
            if (extension == "dat" or extension == "txt"):
                print("Converting {} to hdf...".format(filename))
                self.to_hdf(data_path + filename, sep_regex, filename)


    @staticmethod
    def create_mask(ds,selected_colums,select=True):
            return [select if i in selected_colums else not select for i in range(ds.shape[1])]

    @staticmethod
    def to_hdf(data_path, sep_regex, new_name, header=None):
        """
        This function allows you to read a dat file and store its info in a hdf file.
        :param data_path: rirectory where dat file is located.
        :param sep_regex: regular expression that describe row elements separator in an example (row).
        :param new_name: name of hdf file that will be generated
        :param header: whether file contains header at first row or not.
        :return: None
        """
        split_name = new_name.split(".")
        name_parts = len(split_name)
        if name_parts == 1:
            new_name = new_name + ".hdf"
        elif name_parts == 2:
            new_name = split_name[0] + ".hdf"
        else:
            new_name = "_".join(split_name[:-1]) + ".hdf"

        df = pd.read_csv(data_path, sep=sep_regex, header=header)
        df.to_hdf("/".join(data_path.split("/")[:-1]) + "/" + new_name, "data", mode='w')

    @staticmethod
    def filter_columns(ds, mask):
        """
        This function allows you filter colums via boolean mask.
        :param ds: pandas dataset whose columns will be filtered.
        :param mask: list of boolean that indicates whether column is desired or not.
        :return: filtered dataset
        """
        ncol = ds.shape[1]
        if isinstance(mask, list):
            if len(mask) == ncol:
                return ds.loc[:, mask]
            else:
                raise ValueError(
                    "Mask parameter must be a list of boolean with the same number of columns that ds parameter: {}".format(
                        ncol))
        else:
            raise TypeError("Mask parameter must be a list of boolean.")

    @staticmethod
    def read_data(data_path, verbose=1, filename_separator="-", header=None, formats=["hdf"], **kwargs):
        """
        This function allows you read hdf files into a folder, filtering via kwargs.

        :param data_path: Folder where hdf files to read are.
        :param verbose: If you want read additional info about the process.
        :param filename_separator: Which character separate words in each filename.
        :param kwargs: Additional arguments in order to filter what files to read.
        :return: Dictionary containing files information as pandas dataframe. Additionally, file names are also returned
                 in order to list info without knowing file names.
        """

        files = os.listdir(data_path)

        # remove extension
        split_files = [file.split(".") for file in files]

        # isolate extension
        extensions = [file[1].lower() for file in split_files]

        # file_names contains filenames without extension
        file_names = [file[0].split(filename_separator) for file in split_files]

        criteria = set([criteria.lower() for k,criteria in kwargs.items() if k not in ["formats"] ])



        # Intersection of coincidences
        selected_indexes = np.array([
            i for i, name in enumerate(file_names)
            if set([
                (word.str()).lower() if not isinstance(word, str) else word.lower() for word in name
            ]).intersection(criteria) == criteria and extensions[i] in formats
        ])


        if len(selected_indexes) > 0:
            possible_formats = {
                "hdf": pd.read_hdf,
                "txt":pd.read_csv,
            }
            
            selected_files = np.take(files, selected_indexes)
            selected_extensions = np.take(extensions, selected_indexes)
            #data = {file: pd.read_hdf(data_path + "/" + file, header=header) for file in selected_files}
            data = {file: possible_formats[format](data_path + "/" + file, header=header) for file,format in zip(selected_files, selected_extensions)}

            if verbose:
                for file in selected_files:
                    print('Leido:' + file + ' con shape: ' + str(data[file].shape))

            #for k, v in zip(data.keys(), data.values()):
                #print("-----------------------------")
                #print(k)
                #print(y_index)
                #print(v.shape)
                #print("-----------------------------")
                #y = v.iloc[:, y_calculated_index]
                #del v[v.columns[y_calculated_index]]
                #x = v
                #data[k] = x

            return selected_files, data
        else:
            raise IndexError("There are no files matching the given criteria. Pay attention to \"formats\" argument. "
                             "If you are sure there are, maybe you should convert it to hdf format.")



    @staticmethod
    def split_train_test_events(ds,event_col_idx=0,sim_col_idx=0,test_size=0.2,y_column=0):
        """
        This function allows you to split datasets selectings whole events instead of single rows.
        :param ds: dataset with outputs.
        :param event_col_idx: column that identifies event.
        :param test_size: what portion of data will be used by test.
        :param y_column: index of column where output variable is located.
        :return: training and test sets with separated outputs.
        """
        events_id = pd.unique(ds[ds.columns[event_col_idx]])
        sims_id = pd.unique(ds[ds.columns[sim_col_idx]])

        eid_idx = np.arange(events_id.shape[0])
        sid_idx = np.arange(sims_id.shape[0])

        test_idx = np.random.choice(eid_idx.shape[0], int(eid_idx.shape[0] * test_size), replace=False)
        train_idx = np.setdiff1d(eid_idx, test_idx)

        train_data = ds.loc[ds[ds.columns[event_col_idx]].isin(events_id[train_idx])]
        test_data = ds.loc[ds[ds.columns[event_col_idx]].isin(events_id[test_idx])]

        #y_col=y_column
        #if(y_col < 0):
        #    y_col=ds.columns.shape[0] - abs(y_col)



        y_train = train_data[train_data.columns[y_column]]
        #X_train = train_data.loc[:, train_data.columns != y_col]
        X_train = train_data.loc[:, train_data.columns != train_data.columns[y_column]]

        y_test = test_data[test_data.columns[y_column]]
        X_test = test_data.loc[:, test_data.columns != test_data.columns[y_column]]


        return X_train, y_train, X_test, y_test

    @staticmethod
    def split_train_test(ds, test_size=0.2, y_column=0):
        examples_idx = np.arange(ds.shape[0])
        test_idx = np.random.choice(examples_idx.shape[0], int(examples_idx.shape[0] * test_size), replace=False)
        train_idx = np.setdiff1d(examples_idx, test_idx)

        #print(test_idx.shape[0]/examples_idx.shape[0])
        #print(train_idx.shape[0] / examples_idx.shape[0])

        train_data = ds.iloc[train_idx]
        test_data = ds.iloc[test_idx]

        #print(train_data.shape[0] / ds.shape[0])
        #print(test_data.shape[0] / ds.shape[0])

        y_train = train_data[train_data.columns[y_column]]
        X_train = train_data.loc[:, train_data.columns != train_data.columns[y_column]]

        #print(X_train.shape[0] / ds.shape[0])
        #print(y_train.shape[0] / ds.shape[0])

        y_test = test_data[test_data.columns[y_column]]
        X_test = test_data.loc[:, test_data.columns != test_data.columns[y_column]]

        #print(X_test.shape[0] / ds.shape[0])
        #print(y_test.shape[0] / ds.shape[0])

        return X_train, y_train, X_test, y_test

# Column selection
# tup=[True, False, True, True,...]
# a.loc[:,tup]

# If we want to get index from column name
# a.columns.get_loc("column_name")
