#!/bin/bash
# inference.sh

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

export HOME=$(pwd)

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME="xtreme"
# if you need the environment directory to be named something other than the environment name, change this line
export ENVDIR=$(ENVNAME)

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir -p $ENVDIR
# Set the PYTHONTZPATH to use absolute paths
export PYTHONTZPATH="$ENVDIR/share/zoneinfo:$HOME/$ENVDIR/share/tzinfo"

# First, copy the tar.gz file from /staging into the working directory,
# and untar it to reveal your large input file(s) or directories:

echo "Copying scripts.tar.gz from /staging ..."
cp /staging/syang662/scripts.tar.gz ./
echo "Copying scripts.tar.gz from /staging done."

echo "Untarring scripts.tar.gz..."
tar -xzvf scripts.tar.gz
echo "Untarring scripts.tar.gz done."

if [ ! -d "$ENVNAME" ]; then
  echo "Copying $ENVNAME.tar.gz from /staging ..."
  cp /staging/syang662/$ENVNAME.tar.gz ./
  echo "Copying $ENVNAME.tar.gz from /staging done."

  echo "Untarring $ENVNAME.tar.gz..."
  tar -xzvf $ENVNAME.tar.gz -C $ENVDIR
  echo "Untarring $ENVNAME.tar.gz done."
  . $ENVDIR/bin/activate
fi

# mkdir -p ~/.cache/huggingface/hub/
# echo "Copying models--google--flan-ul2.tar.gz from /staging ..."
# cp /staging/syang662/models--google--flan-ul2.tar.gz ~/.cache/huggingface/hub/
# echo "Copying models--google--flan-ul2.tar.gz from /staging done."

# echo "Untarring models--google--flan-ul2.tar.gz..."
# tar -xzvf ~/.cache/huggingface/hub/models--google--flan-ul2.tar.gz -C ~/.cache/huggingface/hub/
# echo "Untarring models--google--flan-ul2.tar.gz done."

echo "Running predict.py..."
PREDICT_OUT="predict.out"
python3 scripts/src/predict.py | tee $PREDICT_OUT
echo "Inference done."

get_config_attribute() {
  python3 scripts/src/parameters.py $1
}

# DATASET_FILE=$(get_config_attribute dataset_file)
RESULTS_FILE=$(get_config_attribute results_file)
WARNINGS_FILE=$(get_config_attribute warnings_file)
EXECUTION_REPORT_FILE=$(get_config_attribute execution_report_file)
FORMAT_WARNINGS_FILE=$(get_config_attribute format_warnings_file)
# CHECKPOINT_RESULTS_FILE=$(get_config_attribute checkpoint_results_file)
# CHECKPOINT_WARNINGS_FILE=$(get_config_attribute checkpoint_warnings_file)

WEIGHTS_DIR=$(get_config_attribute finetuned_model_dir)
LOGS_DIR=$(get_config_attribute logging_dir)
RANDOM_SEED_FILE=$(get_config_attribute random_seed_path)

files_to_check=("$PREDICT_OUT" "$RESULTS_FILE" "$EXECUTION_REPORT_FILE" "$WARNINGS_FILE" "$FORMAT_WARNINGS_FILE" "$CHECKPOINT_RESULTS_FILE" "$CHECKPOINT_WARNINGS_FILE" "$WEIGHTS_DIR" "$LOGS_DIR" "$RANDOM_SEED_FILE")

files_to_tar=()

for file in "${files_to_check[@]}"; do
  if [ -e "$file" ]; then
    files_to_tar+=("$file")
  fi
done

# tar and move large output files to staging so they're not copied to the submit server:
echo "Tarring output files..."
tar -czvf scripts_results.tar.gz "${files_to_tar[@]}"
echo "Tarring output files done."

echo "Copying scripts_results.tar.gz to /staging..."
cp scripts_results.tar.gz /staging/syang662/
echo "Copying scripts_results.tar.gz to /staging done."

# Before the script exits, make sure to remove the file(s) from the working directory
rm scripts_results.tar.gz $PREDICT_OUT
rm scripts.tar.gz
rm $ENVNAME.tar.gz
rm -r -f $ENVDIR
rm -r -f dataset
rm -r -f scripts
# rm -r -f ~/.cache/huggingface/
