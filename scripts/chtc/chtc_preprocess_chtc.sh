#!/bin/bash

# chtc_preprocess.sh

# run on chtc

get_config_attribute() {
    python3 scripts/src/parameters.py $1
}

RANDOM_SEED_FILE=$(get_config_attribute random_seed_path)

# scripts.tar.gz
SCRIPTS_TAR="scripts.tar.gz"
# if [ ! -f "$SCRIPTS_TAR" ]; then
if true; then
    tar -czvf $SCRIPTS_TAR $RANDOM_SEED_FILE scripts/src/parameters.py scripts/src/predict.py scripts/src/utils.py cache/
else
    echo "$SCRIPTS_TAR already exists. Skipping compression."
fi

ENVNAME="xtreme"
ENV_TAR_GZ="$ENVNAME.tar.gz"
if [ ! -f "$ENV_TAR_GZ" ]; then
    conda pack -n $ENVNAME --dest-prefix='$ENVDIR'
    chmod 644 $ENVNAME.tar.gz
    ls -sh $ENVNAME.tar.gz
else
    echo "$ENV_TAR_GZ already exists. Skipping compression."
fi

# transmit files to /staging
mv scripts.tar.gz /staging/syang662
# scp $ENVNAME.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662
