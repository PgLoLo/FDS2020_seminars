import argparse
from pathlib import Path
from typing import Union, List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=Path, default=Path('./data/nycflights'))
    parser.add_argument('--test-size', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=19)

    return parser.parse_args()


def main(args):
    if args.backend == 'pandas':
        import pandas
        backend = pandas
    elif args.backend == 'dask':
        import dask.dataframe
        backend = dask.dataframe
    else:
        raise ValueError(f'unknown backend: "{args.backend}"')

    data_loader = DataLoader(backend)
    data_train, data_test = data_loader.__load_train_test(args.data, args.test_size, args.seed)

    df_to_Xy = DataframeToXy()
    X_train, y_train = df_to_Xy.__transform(data_train)
    X_test, y_test = df_to_Xy.__transform(data_test)

    param_grid = {
        'learning_rate': [0.05, 0.1, 0.5],
        'n_estimators': 2 ** np.arange(6, 11),
        'subsample': [0.5, 0.75, 1.],
        'min_samples_leaf': 2 ** np.arange(1, 4),
        'max_depth': [2, 3, 5],
        'max_features': ['sqrt', 'log2'],
    }

    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        n_jobs=24,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    print(grid_search)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    predicted = grid_search.predict(X_test)
    print(((predicted - y_test)**2).mean())


if __name__ == '__main__':
    main(get_arguments())
