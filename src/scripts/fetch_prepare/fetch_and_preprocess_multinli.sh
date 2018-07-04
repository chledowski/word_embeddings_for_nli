#!/bin/bash -x
# Preprocessing of SNLI dataset to h5 files, vocab.txt and vocab_all.txt
# Assumes SNLI is downloaded to $DATA_DIR/raw/snli_1.0

set -x

if [ -z "$DATA_DIR" ]; then
    echo "Need to set $DATA_DIR"
    exit 1
fi

wget -P $DATA_DIR/raw/ http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip

mkdir $DATA_DIR/raw/multinli_1.0/
mkdir $DATA_DIR/mnli/
unzip $DATA_DIR/raw/multinli_1.0.zip -d $DATA_DIR/raw/multinli_1.0/
rm $DATA_DIR/raw/multinli_1.0.zip

 Convert to h5 files
python src/util/pack_to_hdf5.py $DATA_DIR/raw/multinli_1.0/multinli_1.0/multinli_1.0_train.txt $DATA_DIR/mnli/train.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.txt $DATA_DIR/mnli/dev.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.txt $DATA_DIR/mnli/dev_mismatched.h5 --type=snli

 Build vocab for both train and all data
python src/util/build_vocab.py $DATA_DIR/mnli/train.h5 $DATA_DIR/mnli/vocab.txt
python src/util/build_vocab.py $DATA_DIR/mnli/train.h5,$DATA_DIR/mnli/dev.h5,$DATA_DIR/mnli/dev_mismatched.h5 $DATA_DIR/mnli/vocab_all.txt
