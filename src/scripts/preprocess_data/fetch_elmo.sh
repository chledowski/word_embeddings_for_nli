#!/bin/bash -x

set -x

if [ -z "$DATA_DIR" ]; then
    echo "Need to set $DATA_DIR"
    exit 1
fi

mkdir -p $DATA_DIR/elmo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
mv elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 ${DATA_DIR}/elmo/weights.h5
mv elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json${DATA_DIR}/elmo/options.json

