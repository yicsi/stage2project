'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pandas as pd
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        file_path = self.dataset_source_folder_path + self.dataset_source_file_name

        df = pd.read_csv(file_path, header=None)

        # first column is label, remaining 784 columns are pixel values
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        # normalize pixel values to [0, 1]
        X = X.astype('float32') / 255.0
        y = y.astype('int64')

        return {'X': X, 'y': y}