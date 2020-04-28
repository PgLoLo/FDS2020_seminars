import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from ml_utils import create_param_grid, Timer
from xy_loader import XyLoader


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=Path, default=Path('./data/nycflights'))
    parser.add_argument('--test-size', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--shrink-factor', type=float, default=.5)

    return parser.parse_args()


def perform_grid_search(X, y, n_jobs):
    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=create_param_grid(),
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=10,
    )

    grid_search.fit(X, y)
    return grid_search


def train_test_final_model(X_train, y_train, X_test, y_test, parameters):
    regressor = GradientBoostingRegressor(**parameters)
    regressor.fit(X_train, y_train)
    return mean_squared_error(y_test, regressor.predict(X_test))


def main(args):
    with Timer() as data_load_timer:
        xy_loader = XyLoader(pd, args.data, args.test_size, args.seed, shrink=args.shrink_factor)
        (X_train, y_train), (X_test, y_test) = xy_loader.load()

    with Timer() as grid_search_timer:
        grid_search = perform_grid_search(X_train, y_train, args.n_jobs)

    with Timer() as final_model_timer:
        final_loss = train_test_final_model(X_train, y_train, X_test, y_test, grid_search.best_params_)

    print(
        f'Data loading time:        {data_load_timer.elapsed_time:7.3f}\n'
        f'Grid search time:         {grid_search_timer.elapsed_time:7.3f}\n'
        f'Final model testing time: {final_model_timer.elapsed_time:7.3f}\n'
        f'Final loss: {final_loss}\n'
    )


if __name__ == '__main__':
    main(get_arguments())
