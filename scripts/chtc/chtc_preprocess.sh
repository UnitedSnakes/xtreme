#!/bin/bash

# chtc_preprocess.sh

# run on sjmc

get_config_attribute() {
    python3 scripts/src/parameters.py $1
}

# scripts.tar.gz
SCRIPTS_TAR="scripts.tar.gz"
# if [ ! -f "$SCRIPTS_TAR" ]; then
if true; then
    DATASET_FILE=$(get_config_attribute dataset_file)
    tar -czvf $SCRIPTS_TAR $DATASET_FILE scripts/src/parameters.py scripts/src/predict.py scripts/src/utils.py
else
    echo "$SCRIPTS_TAR already exists. Skipping compression."
fi

ENVNAME="xtreme"
ENV_TAR_GZ="$ENVNAME.tar.gz"
if [ ! -f "$ENV_TAR_GZ" ]; then
    # ENVNAME=llm
    # conda pack -n $ENVNAME -o $ENVNAME.tar
    # gzip -1 $ENVNAME.tar
    # chmod 644 $ENVNAME.tar.gz
    conda pack -n $ENVNAME --dest-prefix='$ENVDIR'
    chmod 644 $ENVNAME.tar.gz
    ls -sh $ENVNAME.tar.gz
else
    echo "$LLM_TAR_GZ already exists. Skipping compression."
fi

# models--google--flan-ul2.tar.gz
# FLAN_TAR_GZ="models--google--flan-ul2.tar.gz"
# if [ ! -f "$FLAN_TAR_GZ" ]; then
#     tar -czvf $FLAN_TAR_GZ ~/.cache/huggingface/hub/models--google--flan-ul2
# else
#     echo "$FLAN_TAR_GZ already exists. Skipping compression."
# fi

# transmit files to chtc
scp scripts.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662
scp $ENVNAME.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662
# scp models--google--flan-ul2.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662
