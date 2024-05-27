#!/bin/bash

# chtc_preprocess.sh

# run on sjmc

get_config_attribute() {
    python3 scripts/Shanglin/Varsha/src/parameters.py $1
}

# content.tar.gz
CONTENT_TAR="content.tar.gz"
# if [ ! -f "$CONTENT_TAR" ]; then
if true; then
    DATASET_FILE=$(get_config_attribute dataset_file)
    tar -czvf $CONTENT_TAR $DATASET_FILE scripts/Shanglin/Varsha/src/parameters.py scripts/Shanglin/Varsha/src/predict.py scripts/Shanglin/Varsha/src/utils.py
else
    echo "$CONTENT_TAR already exists. Skipping compression."
fi

# llm.tar.gz
LLM_TAR_GZ="llm.tar.gz"
if [ ! -f "$LLM_TAR_GZ" ]; then
    ENVNAME=llm
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
FLAN_TAR_GZ="models--google--flan-ul2.tar.gz"
if [ ! -f "$FLAN_TAR_GZ" ]; then
    tar -czvf $FLAN_TAR_GZ ~/.cache/huggingface/hub/models--google--flan-ul2
else
    echo "$FLAN_TAR_GZ already exists. Skipping compression."
fi

# transmit files to chtc
scp content.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662
scp llm.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662/llm.tar.gz
# scp models--google--flan-ul2.tar.gz syang662@ap2002.chtc.wisc.edu:/staging/syang662
