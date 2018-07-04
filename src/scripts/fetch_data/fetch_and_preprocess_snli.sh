#!/bin/bash -x
# Preprocessing of SNLI dataset to h5 files, vocab.txt and vocab_all.txt
# Assumes SNLI is downloaded to $DATA_DIR/raw/snli_1.0

set -x

if [ -z "$DATA_DIR" ]; then
    echo "Need to set $DATA_DIR"
    exit 1
fi

# Download SNLI
python src/scripts/fetch_data/get_snli.py

# Convert to h5 files
python src/util/pack_to_hdf5.py /home/kchledowski/Breaking_NLI/data/dataset.jsonl $DATA_DIR/snli/test_breaking_nli.h5 --type=snli --breaking_nli_dataset
python src/util/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_train.txt $DATA_DIR/snli/train.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_dev.txt $DATA_DIR/snli/dev.h5 --type=snli
python src/util/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_test.txt $DATA_DIR/snli/test.h5 --type=snli

# Build vocab for both train and all data
python src/util/build_vocab.py $DATA_DIR/snli/train.h5 $DATA_DIR/snli/vocab.txt
python src/util/build_vocab.py $DATA_DIR/snli/train.h5,$DATA_DIR/snli/dev.h5,$DATA_DIR/snli/test.h5 $DATA_DIR/snli/vocab_all.txt
