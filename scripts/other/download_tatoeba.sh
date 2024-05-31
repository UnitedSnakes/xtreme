function download_tatoeba {
    base_dir=$DIR/tatoeba-tmp/
    wget https://github.com/facebookresearch/LASER/archive/main.zip
    unzip -qq -o main.zip -d $base_dir/
    mv $base_dir/LASER-main/data/tatoeba/v1/* $base_dir/
    python $REPO/utils_preprocess.py \
      --data_dir $base_dir \
      --output_dir $DIR/tatoeba \
      --task tatoeba
    rm -rf $base_dir main.zip
    echo "Successfully downloaded data at $DIR/tatoeba" >> $DIR/download.log
}