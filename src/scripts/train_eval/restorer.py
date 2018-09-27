#!/usr/bin/env pythonpl
# -*- coding: utf-8 -*-q
"""
Simple model definitions
"""

import argparse
import json
import logging
import os
import pickle
import shutil

import keras.backend as K
import lmdb
from keras.layers import Dense
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import RandomState
from numpy.random import seed
from tensorflow import set_random_seed

from src import DATA_DIR
from src.models import build_model
from src.scripts.train_eval.utils import build_data_and_streams
from src.util.prepare_embedding import prep_embedding_matrix

logger = logging.getLogger(__name__)


def _serialize_pickle(a):
    return pickle.dumps(a)


def _deserialize_pickle(serialized):
    return pickle.loads(serialized)


def eval_model():
    with open(os.path.join('results', args.model_name, 'config.json'), 'r') as f:
        config = json.load(f)

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=["snli"])

    data = datasets[config["dataset"]]
    model = build_model(config, data)

    embedding_matrix = prep_embedding_matrix(config, data, config["embedding_name"])

    losses = []

    for epoch in range(1, 2):
        model_path = os.path.join('results', args.model_name, f'model_{epoch:02d}.h5')
        print(model_path)
        if not os.path.exists(model_path):
            break

        model.load_weights(os.path.join('results', args.model_name, "best_model.h5"))

        lstm_model = Model(inputs=model.input, outputs=[model.get_layer(args.bilstm_name).get_output_at(i) for i in range(2)])
        print(model.get_layer(args.bilstm_name).get_output_shape_at(0))
        lmdb_path = os.path.join(DATA_DIR, args.lmdb_dir)

        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.makedirs(lmdb_path)

        env = lmdb.open(lmdb_path)
        txn = env.begin(write=True)
        sample_id = 0

        for x in streams["snli"]["train"]:
            input, output = x
            lstm_embeddings = lstm_model.predict(input)
            print(lstm_embeddings[0].shape)
            # print(lstm_embeddings.shape)

            for i, batch_sentence_embeddings in enumerate(lstm_embeddings):

                for j, sentence_embeddings in enumerate(batch_sentence_embeddings):
                    print(len(sentence_embeddings))
                    for k, word_embedding in enumerate(sentence_embeddings):
                        word_id = input[2 * i][j, k]
                        if 0 < word_id < args.top_k:
                            txn.put((str(sample_id) + '_' + str(word_id)).encode(encoding='UTF-8'), _serialize_pickle(word_embedding))
                            sample_id += 1
            txn.commit()
            txn = env.begin(write=True)
        env.close()

        embed = Embedding(data.vocab.size(), config["embedding_dim"],
                          weights=[embedding_matrix],
                          input_length=config["sentence_max_length"],
                          trainable=False,
                          mask_zero=False)

        lstm_embeddings = Input(shape=(600,))
        hidden_layer = Dense(activation='relu')(lstm_embeddings)
        hidden_layer_bn = BatchNormalization()(hidden_layer)
        output = Dense(300)(hidden_layer_bn)

        restorer_model = Model(inputs=[lstm_embeddings], outputs=[output])

        def mean_squared_error_embed(y_true, y_pred):
            y_true = embed(y_true)
            return K.mean(K.square(y_pred - y_true), axis=-1)

        restorer_model.compile(loss=mean_squared_error_embed,
                              optimizer=Adam(),
                              metrics=['loss'])

        def generator():
            env = lmdb.open(lmdb_path, readonly=True, max_readers=2048, max_spare_txns=4)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    word_id = key.decode().split('_')[1]
                    vector = _deserialize_pickle(value)
                    yield [word_id, vector]
                cursor.close()

        model.fit_generator(generator, epochs=10, verbose=True)
        losses.append(model.evaluate_generator(generator))

        print(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bilstm-name", type=str, default='bidirectional_1')
    parser.add_argument("--lmdb-dir", type=str, default='results/lmdb')
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--top-k", type=int, default=10000, help='Take top k words from dict')

    args = parser.parse_args()
    eval_model()