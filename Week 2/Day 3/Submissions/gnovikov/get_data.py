import argparse
from pathlib import Path

import tarfile
import urllib.request


def flights(url: str, path: Path):
    flights_raw = path / 'nycflights.tar.gz'
    flightdir = path / 'nycflights'

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

    print("** Finished! **")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url', type=str, default='https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz'
    )
    parser.add_argument('--path', type=Path, default=Path('./data'))

    return parser.parse_args()


def main(args):
    print("Setting up data directory")
    print("-------------------------")

    flights(args.url, args.path)

    print('Finished!')


if __name__ == '__main__':
    main(get_arguments())
