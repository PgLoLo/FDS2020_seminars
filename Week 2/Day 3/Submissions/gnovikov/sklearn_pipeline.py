import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from xy_loader import XyLoader


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=Path, default=Path('./data/nycflights'))
    parser.add_argument('--test-size', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--shrink-factor', type=float, default=.5)

    return parser.parse_args()


def create_param_grid():
    return {
        'learning_rate': [0.05, 0.5],
        'n_estimators': 2 ** np.arange(6, 10),
        'subsample': [0.5, 1.],
        'min_samples_leaf': 2 ** np.arange(1, 4),
        'max_depth': [3, 5],
        'max_features': ['sqrt', 'log2'],
    }


def main(args):
    start_time = time.time()

    xy_loader = XyLoader(pd, args.data, args.test_size, args.seed, shrink=args.shrink_factor)
    (X_train, y_train), (X_test, y_test) = xy_loader.load()
    data_loaded = time.time()

    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=create_param_grid(),
        scoring='neg_mean_squared_error',
        n_jobs=args.n_jobs,
        verbose=10,
    )

    grid_search.fit(X_train, y_train)
    grid_search_finished = time.time()

    print(grid_search)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    predicted = grid_search.predict(X_test)
    print(((predicted - y_test)**2).mean())
    print(
        f'Data loading time: {data_loaded - start_time:.3f}\n'
        f'Grid searcg time:  {grid_search_finished - data_loaded}\n'
        f'Total:             {grid_search_finished - start_time}'
    )


if __name__ == '__main__':
    main(get_arguments())
