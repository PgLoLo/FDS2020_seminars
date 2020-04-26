import argparse
from pathlib import Path
from typing import Union, List

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataframes(csv_list: List[Path]):
    dataframes = [pd.read_csv(csv) for csv in csv_list]
    return pd.concat(dataframes)


def load_data(path: Path, test_size: Union[int, float]):
    csvs = list(path.glob('*.csv'))
    train, test = train_test_split(csvs, test_size=test_size)

    return load_dataframes(train), load_dataframes(test)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=Path, default=Path('./data/nycflights'))
    parser.add_argument('--test-size', type=float, default=.3)

    return parser.parse_args()


def main(args):
    data_train, data_test = load_data(args.data, args.test_size)
    print(data_train.info())
    print(data_test.info())


if __name__ == '__main__':
    main(get_arguments())
