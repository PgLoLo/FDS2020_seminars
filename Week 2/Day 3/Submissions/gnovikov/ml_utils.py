import time

import numpy as np


def create_param_grid():
    return {
        'learning_rate': [0.05, 0.5],
        'n_estimators': 2 ** np.arange(7, 10),
        'subsample': [0.5, 1.],
        'min_samples_leaf': [4, 8],
        'max_depth': [3, 5],
        'max_features': ['sqrt', 'log2'],
    }


class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
