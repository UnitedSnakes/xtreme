#!/bin/bash
# inference.sh

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=llm
# if you need the environment directory to be named something other than the environment name, change this line
export ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir -p $ENVDIR
# Set the PYTHONTZPATH to use absolute paths
export PYTHONTZPATH="$ENVDIR/share/zoneinfo:$ENVDIR/share/tzinfo"

# First, copy the tar.gz file from /staging into the working directory,
# and untar it to reveal your large input file(s) or directories:

echo "Copying content.tar.gz from /staging ..."
cp /staging/syang662/content.tar.gz ./
echo "Copying content.tar.gz from /staging done."

echo "Untarring content.tar.gz..."
tar -xzvf content.tar.gz
echo "Untarring content.tar.gz done."

echo "Copying $ENVNAME.tar.gz from /staging ..."
cp /staging/syang662/$ENVNAME.tar.gz ./
echo "Copying $ENVNAME.tar.gz from /staging done."

echo "Untarring $ENVNAME.tar.gz..."
tar -xzvf $ENVNAME.tar.gz -C $ENVDIR
echo "Untarring $ENVNAME.tar.gz done."
. $ENVDIR/bin/activate

# mkdir -p ~/.cache/huggingface/hub/
# echo "Copying models--google--flan-ul2.tar.gz from /staging ..."
# cp /staging/syang662/models--google--flan-ul2.tar.gz ~/.cache/huggingface/hub/
# echo "Copying models--google--flan-ul2.tar.gz from /staging done."

# echo "Untarring models--google--flan-ul2.tar.gz..."
# tar -xzvf ~/.cache/huggingface/hub/models--google--flan-ul2.tar.gz -C ~/.cache/huggingface/hub/
# echo "Untarring models--google--flan-ul2.tar.gz done."

echo "Running inference..."
python3 scripts/Shanglin/Varsha/src/predict.py -no_start_from_cp_file -overwrite | tee scripts/Shanglin/Varsha/src/predict.out
echo "Inference done."

get_config_attribute() {
  python3 scripts/Shanglin/Varsha/src/parameters.py $1
}

PREDICT_OUT="scripts/Shanglin/Varsha/src/predict.out"
DATASET_FILE=$(get_config_attribute dataset_file)
RESULTS_FILE=$(get_config_attribute results_file)
WARNINGS_FILE=$(get_config_attribute warnings_file)
EXECUTION_REPORT_FILE=$(get_config_attribute execution_report_file)
FORMAT_WARNINGS_FILE=$(get_config_attribute format_warnings_file)
CHECKPOINT_RESULTS_FILE=$(get_config_attribute checkpoint_results_file)
CHECKPOINT_WARNINGS_FILE=$(get_config_attribute checkpoint_warnings_file)

files_to_tar=("$PREDICT_OUT" "$RESULTS_FILE" "$EXECUTION_REPORT_FILE")

if [ -f "$WARNINGS_FILE" ]; then
  files_to_tar+=("$WARNINGS_FILE")
fi

if [ -f "$FORMAT_WARNINGS_FILE" ]; then
  files_to_tar+=("$FORMAT_WARNINGS_FILE")
fi

if [ -f "$CHECKPOINT_RESULTS_FILE" ]; then
  files_to_tar+=("$CHECKPOINT_RESULTS_FILE")
fi

if [ -f "$CHECKPOINT_WARNINGS_FILE" ]; then
  files_to_tar+=("$CHECKPOINT_WARNINGS_FILE")
fi

# tar and move large output files to staging so they're not copied to the submit server:
echo "Tarring output files..."
tar -czvf content_results.tar.gz "${files_to_tar[@]}"
echo "Tarring output files done."

echo "Copying content_results.tar.gz to /staging..."
cp content_results.tar.gz /staging/syang662/
echo "Copying content_results.tar.gz to /staging done."

# Before the script exits, make sure to remove the file(s) from the working directory
rm content_results.tar.gz scripts/Shanglin/Varsha/src/predict.out
rm content.tar.gz
rm $ENVNAME.tar.gz
rm -r -f $ENVDIR
rm -r -f dataset
rm -r -f scripts
# rm -r -f ~/.cache/huggingface/
