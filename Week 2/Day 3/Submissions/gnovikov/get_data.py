from __future__ import print_function

import os

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob


def flights(url: str, n_rows: Optional[int], path: Path):
    flights_raw = path / 'nycflights.tar.gz'
    flightdir = path / 'nycflights'
    jsondir = path / 'flightjson'

    if not path.exists():
        path.mkdir()

    if not flights_raw.exists():
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not flightdir.exists():
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = path / 'nycflights.tar.gz'
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall(path)
        print("done", flush=True)

    if not jsondir.exists():
        print("- Creating json data... ", end='', flush=True)
        jsondir.mkdir()

        for path in flightdir.glob('*.csv'):
            df = pd.read_csv(path)
            if n_rows is not None:
                df = df.iloc[:n_rows]
            df.to_json(jsondir / f'{path.stem}.json', orient='records', lines=True)
        print("done", flush=True)

    print("** Finished! **")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url', type=str, default='https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz'
    )
    parser.add_argument('--n-rows', type=int, default=None)
    parser.add_argument('--path', type=Path, default=Path('./data'))

    return parser.parse_args()


def main(args):
    print("Setting up data directory")
    print("-------------------------")

    flights(args.url, args.n_rows, args.path)

    print('Finished!')


if __name__ == '__main__':
    main(get_arguments())
