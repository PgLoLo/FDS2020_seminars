#!/bin/bash

set -e
set -x

DATADIR="./data"
DATA_LOADER_SCRIPT="get_data.py"
ML_PIPELINE_SCRIPT="$1_pipeline.py"
LOG_FILE="sklearn.log"

python $DATA_LOADER_SCRIPT --path $DATADIR
python $ML_PIPELINE_SCRIPT --data $DATADIR/nycflights > $LOG_FILE