from pathlib import Path
from random import random
from typing import List, Union

from sklearn.model_selection import train_test_split


class XyLoader:
    FEATURES = [
        'CRSDepTime',
        'CRSArrTime',
        'CRSElapsedTime',
        'Distance',
    ]

    TARGET = 'DepDelay'

    def __init__(self, backend, path, test_size, seed, *, shrink=None):
        self.backend = backend
        self.path = path
        self.test_size = test_size
        self.seed = seed
        self.shrink = shrink

    def __skiprows(self, ind):
        if ind == 0 or self.shrink is None:
            return False
        else:
            return random() > self.shrink

    def __load_dataframes(self, csv_list: List[Path]):
        skiprows = self.__skiprows if self.shrink is not None else None
        dataframes = [self.backend.read_csv(csv, skiprows=skiprows) for csv in csv_list]
        return self.backend.concat(dataframes)

    def __load_train_test(self):
        csvs = list(self.path.glob('*.csv'))
        train, test = train_test_split(csvs, test_size=self.test_size, random_state=self.seed)

        return self.__load_dataframes(train), self.__load_dataframes(test)

    def __transform(self, data):
        data = data[self.FEATURES + [self.TARGET]]
        data = data.dropna()
        return data[self.FEATURES], data[self.TARGET]

    def load(self):
        data_train, data_test = self.__load_train_test()
        X_train, y_train = self.__transform(data_train)
        X_test, y_test = self.__transform(data_test)

        return (X_train, y_train), (X_test, y_test)




