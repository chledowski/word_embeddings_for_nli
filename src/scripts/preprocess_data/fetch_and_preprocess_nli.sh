#!/bin/bash -x
# Preprocessing of SNLI dataset to h5 files, vocab.txt and vocab_all.txt
# Assumes SNLI is downloaded to $DATA_DIR/raw/snli_1.0

set -x

if [ -z "$DATA_DIR" ]; then
    echo "Need to set $DATA_DIR"
    exit 1
fi

############################## Breaking NLI

wget --no-check-certificate --content-disposition https://github.com/BIU-NLP/Breaking_NLI/blob/master/breaking_nli_dataset.zip?raw=true
unzip breaking_nli_dataset.zip -d breaking_nli_dataset
mv breaking_nli_dataset/data/dataset.jsonl $DATA_DIR/raw/test_breaking_nli.jsonl
rm -rf breaking_nli_dataset
rm breaking_nli_dataset.zip

python src/util/pack_to_hdf5.py $DATA_DIR/raw/test_breaking_nli.jsonl $DATA_DIR/snli/test_breaking_nli.h5 --type=snli --breaking_nli_dataset
python src/util/convert_breaking_to_txt.py $DATA_DIR/snli/test_breaking_nli.h5

############################## SNLI

# Download SNLI
python src/scripts/fetch_data/get_snli.py

# Convert to h5 files
python src/util/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_train.txt $DATA_DIR/snli/train.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_dev.txt $DATA_DIR/snli/dev.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_test.txt $DATA_DIR/snli/test.h5 --type=snli

# Build vocab for both train and all data
python src/util/build_vocab.py $DATA_DIR/snli/train.h5 $DATA_DIR/snli/vocab.txt
python src/util/build_vocab.py $DATA_DIR/snli/train.h5,$DATA_DIR/snli/dev.h5,$DATA_DIR/snli/test.h5 $DATA_DIR/snli/vocab_all.txt

############################## MNLI

if [ ! -d "$DATA_DIR/mnli" ]; then
    wget -P $DATA_DIR/raw/ http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip

    mkdir $DATA_DIR/raw/multinli_1.0/
    mkdir $DATA_DIR/mnli/
    unzip $DATA_DIR/raw/multinli_1.0.zip -d $DATA_DIR/raw/multinli_1.0/
    rm $DATA_DIR/raw/multinli_1.0.zip
else
    echo "$DATA_DIR/mnli exists. Skipping download..."
fi

# Convert to h5 files
python src/util/pack_to_hdf5.py $DATA_DIR/raw/multinli_1.0/multinli_1.0/multinli_1.0_train.txt $DATA_DIR/mnli/train.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.txt $DATA_DIR/mnli/dev.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.txt $DATA_DIR/mnli/dev_mismatched.h5 --type=snli

# Build vocab for both train and all data
python src/util/build_vocab.py $DATA_DIR/mnli/train.h5 $DATA_DIR/mnli/vocab.txt
python src/util/build_vocab.py $DATA_DIR/mnli/train.h5,$DATA_DIR/mnli/dev.h5,$DATA_DIR/mnli/dev_mismatched.h5 $DATA_DIR/mnli/vocab_all.txt