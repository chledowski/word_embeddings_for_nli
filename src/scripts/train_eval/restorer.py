#!/usr/bin/env pythonpl
# -*- coding: utf-8 -*-q
"""
Simple model definitions
"""

import argparse
import json
import logging
import numpy as np
import os
import scipy
import sys
import time
import tqdm
import web.evaluate

import keras.backend as K

from collections import Counter
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error
from keras.layers import Dense
from keras.layers import Input, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.utils import Sequence
from numpy.random import RandomState
from numpy.random import seed
from pprint import pprint
from tensorflow import set_random_seed
from web.evaluate import evaluate_similarity
from web.evaluate import fetch_WS353, fetch_SimLex999, fetch_MEN

from src import DATA_DIR
from src.configs.configs import baseline_configs
from src.models import build_model
from src.scripts.train_eval.utils import build_data_and_streams
from src.util.prepare_embedding import prep_embedding_matrix, norm_weight

logger = logging.getLogger(__name__)

# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
# pydevd.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

seed(0)
set_random_seed(0)
rng = RandomState(0)


def evaluate_similarity(w, X, y):
    all_vectors = []
    for k, vs in w.items():
        all_vectors.extend(vs)

    found_words = []
    missing_words = []
    for word in X[:, 0].tolist() + X[:, 1].tolist():
        if word not in w:
            missing_words.append(word)
        else:
            found_words.append(word)

    print("Missing words", len(missing_words), missing_words[:5])
    print("Found words", len(found_words), found_words[:5])

    mean_vector = np.mean(all_vectors, axis=0)
    scores = []

    tr = tqdm.tqdm(total=len(X))
    for word_a, word_b in X:
        pair_scores = []
        for v1 in w.get(word_a, [mean_vector]):
            for v2 in w.get(word_b, [mean_vector]):
                pair_scores.append(v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        scores.append(np.mean(pair_scores))
        tr.set_postfix({'scores': len(pair_scores)})
        tr.update(1)
    return scipy.stats.spearmanr(scores, y).correlation


def words_in_similarity_tasks():
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "SimLex999": fetch_SimLex999()
    }
    words = set()
    for name, data in similarity_tasks.items():
        for word in data.X[:, 0].tolist() + data.X[:, 1].tolist():
            words.add(word.lower())
    return words


def evaluate_on_similarity_tasks(w):
    # Calculate results on similarity
    logger.info("Calculating similarity benchmarks")
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "SimLex999": fetch_SimLex999()
    }

    similarity_results = {}

    for name, data in similarity_tasks.items():
        similarity_results[name] = evaluate_similarity(w, data.X, data.y)
        print("Spearman correlation of scores on {} {}".format(name, similarity_results[name]))
    return similarity_results


def build_restorer_model(config, data, linear_only=False):
    print("Building restorer model...")

    first_embeddings_input = Input(shape=(600,))
    second_embeddings_input = Input(shape=(600,))

    first_embeddings = first_embeddings_input
    second_embeddings = second_embeddings_input

    hidden_layers = []
    if linear_only:
        linear_layer = Dense(300, name='final_transform', use_bias=False)
        hidden_layers.append(linear_layer)
    else:
        hidden_layers.append(Dense(600, activation='relu', kernel_regularizer=l2()))
        hidden_layers.append(BatchNormalization())
        hidden_layers.append(Dense(300, name='final_transform', kernel_regularizer=l2()))

    for hidden_layer in hidden_layers:
        first_embeddings = hidden_layer(first_embeddings)
        second_embeddings = hidden_layer(second_embeddings)

    output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=1))([first_embeddings, second_embeddings])

    restorer_model = Model(inputs=[first_embeddings_input,
                                   second_embeddings_input],
                           outputs=[output])

    restorer_model.compile(loss=mean_squared_error,
                           optimizer=SGD(1e-3, clipnorm=1.0))

    print("Built.")
    return restorer_model


class RotatorModel:
    def __init__(self, config, embedding_matrix):
        self.ortho_matrix = norm_weight(nin=config["embedding_dim"],
                                        nout=config["embedding_dim"],
                                        ortho=True)
        self.scale_diag = np.array(1.0 + 0*np.random.random(config["embedding_dim"]))[None, :]
        # self.ortho_matrix = self.scale_matrix @ self.ortho_matrix
        self.embedding_matrix = embedding_matrix
        self.scale = config['scale']
        self.rotate = config['rotate']

    def predict(self, input):
        premise, _, hypothesis, _ = input
        input_embeddings = [premise, hypothesis]
        output_embeddings = []
        for i, batch in enumerate(input_embeddings):
            embedded_batch = np.array(
                [self.embedding_matrix[sentence] for sentence in batch]
            )  # [batch_size, num_words, dim]
            if self.rotate:
                embedded_batch = embedded_batch @ self.ortho_matrix
            if self.scale:
                embedded_batch = np.multiply(embedded_batch, self.scale_diag)
            output_embeddings.append(np.concatenate([embedded_batch, embedded_batch], axis=2))
        return np.array(output_embeddings)  # [2, batch_size, num_words, 2*dim]


def load_test_model_and_data(config):
    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=["snli"])
    data = datasets[config["dataset"]]
    stream = streams["snli"]["train"]
    glove_embeddings_matrix = prep_embedding_matrix(config, data, config["embedding_name"])
    model = RotatorModel(config, glove_embeddings_matrix)
    return model, data, stream, glove_embeddings_matrix


def load_model_and_data(config, model_name, layer_name, epoch):
    K.clear_session()

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=["snli"])
    data = datasets[config["dataset"]]
    glove_embeddings_matrix = prep_embedding_matrix(config, data, config["embedding_name"])

    print("Building model...")

    model = build_model(config, data)

    print("Loading weights...")

    weights_path = os.path.join(DATA_DIR, 'results', model_name, f'model_{epoch:02d}.h5')
    model.load_weights(weights_path)

    lstm_model = Model(inputs=model.input,
                       outputs=[model.get_layer(layer_name).get_output_at(i) for i in range(2)])

    return lstm_model, data, streams["snli"]["train"], glove_embeddings_matrix


def dump_embeddings(model, stream, vocab, needed_words, max_num_batches):
    print("Dumping...")

    dumped_embeddings = np.zeros(shape=(args.dump_num_words, 600))
    dumped_ids = np.zeros(args.dump_num_words, dtype=np.int32)
    word_counter = Counter()
    num_dumped = 0
    num_needed_dumped = 0
    num_needed_words = len(needed_words)

    with tqdm.tqdm(total=max_num_batches) as tr:
        for batch_id, x in enumerate(stream):
            input, _ = x
            lstm_embeddings = model.predict(input)  # [2, batch_size, max_len, 2*dim]

            for i, batch_sentence_embeddings in enumerate(lstm_embeddings):
                masks = input[2*i+1]
                for j, sentence_embeddings in enumerate(batch_sentence_embeddings):
                    sentence_len = np.sum(masks[j])
                    for k in range(1, sentence_len-1):  # without <bos> and <eos>
                        word_embedding = sentence_embeddings[k]
                        word_id = input[2 * i][j, k]
                        word = vocab.id_to_word(word_id).decode().lower()

                        if word_counter[word] >= args.dump_max_per_word:
                            # keep only max dump_max_per_word embeddings per word
                            continue
                        if word in needed_words:
                            # replace
                            replaced_word = vocab.id_to_word(dumped_ids[num_needed_dumped]).decode().lower()
                            word_counter[replaced_word] -= 1
                            dumped_ids[num_needed_dumped] = word_id
                            dumped_embeddings[num_needed_dumped] = word_embedding
                            num_needed_dumped += 1
                            num_dumped = max(num_dumped, num_needed_dumped)
                            word_counter[word] += 1
                        elif 6 <= word_id < args.top_k and num_dumped < args.dump_num_words:
                            # append
                            dumped_ids[num_dumped] = word_id
                            dumped_embeddings[num_dumped] = word_embedding
                            num_dumped += 1
                            word_counter[word] += 1
            tr.update(1)
            tr.set_postfix({
                'num_dumped': num_dumped,
                'num_needed_dumped': num_needed_dumped
            })
            if batch_id >= max_num_batches:
                break

    print("Dumped %d embeddings" % num_dumped)
    return dumped_ids[:num_dumped], dumped_embeddings[:num_dumped]


class EmbeddingsSequence(Sequence):
    def __init__(self, ids, embeddings, glove_embeddings_matrix, batch_size):
        self.ids = ids
        self.embeddings = embeddings
        self.glove_embeddings_matrix = glove_embeddings_matrix
        self.batch_size = batch_size

    def __len__(self):
        return self.ids.shape[0] // (self.batch_size * 2)  # *2 because each pair consists of a pair

    def on_epoch_end(self):
        # shuffle
        perm = np.random.permutation(self.ids.shape[0])
        self.ids = self.ids[perm]
        self.embeddings = self.embeddings[perm]

    def __getitem__(self, idx):
        batch_ids = self.ids[2 * idx * self.batch_size:2 * (idx + 1) * self.batch_size]
        batch_embeddings = self.embeddings[2 * idx * self.batch_size:2 * (idx + 1) * self.batch_size]

        first_embeddings = batch_embeddings[:self.batch_size]
        second_embeddings = batch_embeddings[self.batch_size:]

        first_ids = batch_ids[:self.batch_size]
        first_glove_embeddings = self.glove_embeddings_matrix[first_ids]
        second_ids = batch_ids[self.batch_size:]
        second_glove_embeddings = self.glove_embeddings_matrix[second_ids]

        glove_dots = np.sum(first_glove_embeddings * second_glove_embeddings, axis=-1)

        return [first_embeddings, second_embeddings], glove_dots


def train_restorer(restorer_model, ids, embeddings, glove_embeddings_matrix,
                   batch_size, num_epochs, train_percentage):

    num_train_examples = int(train_percentage * ids.shape[0])
    train_sequence = EmbeddingsSequence(ids=ids[:num_train_examples],
                                        embeddings=embeddings[:num_train_examples],
                                        glove_embeddings_matrix=glove_embeddings_matrix,
                                        batch_size=batch_size)
    test_sequence = EmbeddingsSequence(ids=ids[num_train_examples:],
                                       embeddings=embeddings[num_train_examples:],
                                       glove_embeddings_matrix=glove_embeddings_matrix,
                                       batch_size=batch_size)

    restorer_model.fit_generator(generator=train_sequence,
                                 validation_data=test_sequence,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                                 epochs=num_epochs,
                                 verbose=True)

    loss = restorer_model.evaluate_generator(test_sequence,
                                             verbose=True)

    return loss


def prepare_embedding_dict(ids, embeddings, vocab):
    embeddings_dict = {}
    for id, embedding in zip(ids, embeddings):
        word = vocab.id_to_word(id).decode().lower()
        if word not in embeddings_dict:
            embeddings_dict[word] = []
        embeddings_dict[word].append(embedding)

    print("embeddings_dict size", len(embeddings_dict))
    print("embeddings_dict max", np.max([len(vs) for k, vs in embeddings_dict.items()]))
    print("embeddings_dict sample", list(embeddings_dict.keys())[:10])
    return embeddings_dict


def maybe_dump_embeddings(lstm_model, config, data, stream, force_load=False):
    dump_ids_path = os.path.join(
        DATA_DIR, 'results', args.model_name, 'dumped_ids_%d.npy' % args.model_epoch)
    dump_embs_path = os.path.join(
        DATA_DIR, 'results', args.model_name, 'dumped_embs_%d.npy' % args.model_epoch)

    print("Loading embeddings...")

    while force_load and not os.path.exists(dump_ids_path):
        time.sleep(10)

    if force_load:
        dumped_ids = np.load(dump_ids_path)
        dumped_embeddings = np.load(dump_embs_path)
        print("loaded ids:", dumped_ids.shape)
        print("loaded embs:", dumped_embeddings.shape)
    else:
        dumped_ids, dumped_embeddings = dump_embeddings(
            lstm_model,
            stream,
            data.vocab,
            words_in_similarity_tasks(),
            min(args.dump_num_batches, data.num_examples('train') // config['batch_sizes']['snli']['train']))
        np.save(dump_ids_path, dumped_ids)
        np.save(dump_embs_path, dumped_embeddings)
    return dumped_ids, dumped_embeddings


def dump_only(use_test_model):
    if use_test_model:
        # use esim config as default test
        config = dict(baseline_configs['esim'])
        config['scale'] = False
        config['rotate'] = False
        lstm_model, data, stream, glove_embeddings_matrix = load_test_model_and_data(config)
    else:
        with open(os.path.join(DATA_DIR, 'results', args.model_name, 'config.json'), 'r') as f:
            config = json.load(f)
        lstm_model, data, stream, glove_embeddings_matrix = load_model_and_data(
            config, args.model_name, args.layer_name, args.model_epoch
        )
    maybe_dump_embeddings(lstm_model, config, data, stream)


def work(use_test_model):
    if use_test_model:
        # use esim config as default test
        config = dict(baseline_configs['esim'])
        config['scale'] = False
        config['rotate'] = False
        lstm_model, data, stream, glove_embeddings_matrix = load_test_model_and_data(config)
    else:
        with open(os.path.join(DATA_DIR, 'results', args.model_name, 'config.json'), 'r') as f:
            config = json.load(f)

        lstm_model, data, stream, glove_embeddings_matrix = load_model_and_data(
            config, args.model_name, args.layer_name, args.model_epoch
        )

    dumped_ids, dumped_embeddings = maybe_dump_embeddings(lstm_model, config, data, stream, force_load=True)

    if use_test_model:
        assert np.allclose(glove_embeddings_matrix[dumped_ids], dumped_embeddings[:, :300])
        print("Assert OK")

    restorer_model = build_restorer_model(config, data)

    loss = train_restorer(restorer_model,
                          ids=dumped_ids,
                          embeddings=dumped_embeddings,
                          glove_embeddings_matrix=glove_embeddings_matrix,
                          batch_size=args.restorer_batch_size,
                          num_epochs=args.restorer_num_epochs,
                          train_percentage=args.restorer_train_percentage)

    print("Final loss: %.2f" % loss)

    transform_model = Model(inputs=[restorer_model.inputs[0]],
                            outputs=[restorer_model.get_layer('final_transform').get_output_at(0)])

    embeddings = transform_model.predict(dumped_embeddings,
                                         batch_size=args.restorer_batch_size)

    embeddings_dict = prepare_embedding_dict(dumped_ids,
                                             embeddings,
                                             data.vocab)

    results_path = os.path.join(DATA_DIR, 'results/%s/restorer_results.json' % args.model_name)
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            similarity_results = json.load(f)
    else:
        similarity_results = {}

    epoch_results = evaluate_on_similarity_tasks(embeddings_dict)
    epoch_results['loss'] = loss
    similarity_results['epoch_%d' % args.model_epoch] = epoch_results

    with open(results_path, 'w') as f:
        json.dump(similarity_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-name", type=str, default='bilstm')
    parser.add_argument("--cache-dir", type=str, default='lmdb')
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--top-k", type=int, default=100000, help='Take top k words from dict')
    parser.add_argument("--model-epoch", type=int, required=True)
    parser.add_argument("--restorer-num-epochs", type=int, default=200)
    parser.add_argument("--restorer-batch-size", type=int, default=128)
    parser.add_argument("--restorer-train-percentage", type=float, default=0.9)
    parser.add_argument("--dump-num-words", type=int, default=100000)
    parser.add_argument("--dump-num-batches", type=int, default=1000000)
    parser.add_argument("--dump-max-per-word", type=int, default=20)
    parser.add_argument("--dump-only", action='store_true')
    parser.add_argument("--evaluate-only", action='store_true')

    args = parser.parse_args()

    use_test_model = (args.model_epoch == 0)
    if args.dump_only:
        dump_only(use_test_model)
    else:
        work(use_test_model)
