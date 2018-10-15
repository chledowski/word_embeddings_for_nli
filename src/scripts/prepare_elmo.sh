#!/usr/bin/env bash
"""
Script helper that:
a) collects all tokens from SNLI/MNLI
b) create vocab for ELMO
c) fine-tune ELMO
d) produces token embeddings
"""

set -x
set -e

ELMO_DIR=${DATA_DIR}/elmo
SNLI_DIR=${DATA_DIR}/snli
MNLI_DIR=${DATA_DIR}/mnli

TRAIN_PATH=${ELMO_DIR}/all_train.txt
DEV_PATH=${ELMO_DIR}/all_dev.txt
VOCAB_PATH=${ELMO_DIR}/vocab_elmo.txt

declare -a DATASET_DIRS=(
    "${SNLI_DIR}"
    "${MNLI_DIR}"
)

TMP_TRAIN_PATH="/tmp/train.txt"
TMP_DEV_PATH="/tmp/dev.txt"
TMP_VOCAB_PATH="/tmp/vocab.txt"

if [ ! -f ${TRAIN_PATH} ]; then
    # 1. merge all SNLI & MNLI & Breaking tokens into one file
    rm -f /tmp/tokens.txt
    for (( i=0; i<${#DATASET_DIRS[@]}; i++ ))
    do
        FILEPATHS="${DATASET_DIRS[$i]}/*train_token.txt"
        cat ${FILEPATHS} >> ${TMP_TRAIN_PATH}

        FILEPATHS="${DATASET_DIRS[$i]}/*dev_*token.txt"
        cat ${FILEPATHS} >> ${TMP_DEV_PATH}
    done

    # 2. make them unique
    cat ${TMP_TRAIN_PATH} | sort -u > ${TRAIN_PATH}
    cat ${TMP_DEV_PATH} | sort -u > ${DEV_PATH}
fi

# 3. create vocab
if [ ! -f ${VOCAB_PATH} ]; then
    python3 ${SOURCE_DIR}/util/build_vocab.py "${TRAIN_PATH},${DEV_PATH}" ${TMP_VOCAB_PATH}
    echo -e "<S>\n</S>\n<UNK>" > ${TMP_VOCAB_PATH}_prefix
    tail -n +7 ${TMP_VOCAB_PATH} > ${TMP_VOCAB_PATH}_suffix
    cat ${TMP_VOCAB_PATH}_prefix ${TMP_VOCAB_PATH}_suffix | awk '{print $1}' > ${VOCAB_PATH}
fi

# 4. download checkpoint file
CHECKPOINT_DIR=${ELMO_DIR}/checkpoints
if [ ! -d ${CHECKPOINT_DIR} ]; then
    mkdir -p ${CHECKPOINT_DIR}
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/checkpoint -P ${CHECKPOINT_DIR}
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/options.json -P ${CHECKPOINT_DIR}
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.data-00000-of-00001 -P ${CHECKPOINT_DIR}
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.index -P ${CHECKPOINT_DIR}
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/model.ckpt-935588.meta -P ${CHECKPOINT_DIR}
fi

# 5. fine-tune model
NUM_TRAIN_TOKENS=$(wc -w ${TRAIN_PATH} | awk '{print $1}')

#python ${SOURCE_DIR}/models/bilmtf/bin/restart.py \
#    --save_dir=${CHECKPOINT_DIR} \
#    --vocab_file=${VOCAB_PATH} \
#    --train_prefix=${MERGED_DATASETS_PATH} \
#    --n_gpus=1 \
#    --batch_size=256 \
#    --n_train_tokens=${NUM_TRAIN_TOKENS} \
#    --n_epochs=3

# 6. dump weights
python ${SOURCE_DIR}/models/bilmtf/bin/dump_weights.py \
    --save_dir=${CHECKPOINT_DIR} \
    --outfile="${ELMO_DIR}/lm_weights.hdf5"

# 7. dump embeddings
python ${SOURCE_DIR}/models/bilmtf/bin/dump_tokens.py \
    --save_dir=${ELMO_DIR}