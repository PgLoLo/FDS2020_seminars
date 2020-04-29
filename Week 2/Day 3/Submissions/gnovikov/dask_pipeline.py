import argparse
import time
from pathlib import Path

import dask
import dask.dataframe
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import GridSearchCV
from dask_ml.xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from ml_utils import create_param_grid, Timer
from xy_loader import XyLoader


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=Path, default=Path('./data/nycflights'))
    parser.add_argument('--test-size', type=float, default=.3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--n-workers', type=int, default=8)

    return parser.parse_args()


def perform_grid_search(X, y):
    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=create_param_grid(),
        scoring='neg_mean_squared_error',
    )
    return grid_search.fit(X, y)


def train_test_final_model(X_train, y_train, X_test, y_test, parameters):
    regressor = XGBRegressor(**parameters)
    regressor = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return ((y_pred - y_test) ** 2).mean().compute()


def main(args):
    with LocalCluster(n_workers=args.n_workers) as cluster:
        with Client(cluster):
            with Timer() as data_load_timer:
                xy_loader = XyLoader(dask.dataframe, args.data, args.test_size, args.seed)
                (X_train, y_train), (X_test, y_test) = xy_loader.load()

            with Timer() as grid_search_timer:
                grid_search = perform_grid_search(X_train, y_train)

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
