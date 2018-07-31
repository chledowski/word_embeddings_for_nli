import numpy as np
import os
import pickle as pkl

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from src import DATA_DIR
from src.util.data import SNLIData
from src.util.vocab import Vocabulary

import tensorflow as tf


def prepare_kb(config, features, x1_lemma, x2_lemma):
    batch_size = x1_lemma.shape[0]
    # print("KB batch_size: %d" % batch_size)
    kb_x = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')
    kb_y = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')

    total_misses = 0.0
    total_hits = 0.0

    def fill_kb(batch_id, words1, words2, kb):
        hits, misses = 0, 0
        for i1, w1 in enumerate(words1):
            for i2, w2 in enumerate(words2):
                if w1 in features and w2 in features[w1]:
                    kb[batch_id][i1][i2] = features[w1][w2]
                    hits += 1
                else:
                    misses += 1
        return hits, misses

    for batch_id in range(batch_size):
        h, m = fill_kb(batch_id, x1_lemma[batch_id], x2_lemma[batch_id], kb_x)
        total_hits += h
        total_misses += m
        h, m = fill_kb(batch_id, x2_lemma[batch_id], x1_lemma[batch_id], kb_y)
        total_hits += h
        total_misses += m

    print("Hits ratio: %.2f" % (total_hits / (total_hits + total_misses)))
    return kb_x, kb_y, total_hits, total_misses


def load_pair_features(config):
    features = {}
    features_pkl_path = os.path.join(DATA_DIR, config['pair_features_pkl_path'])
    if not os.path.exists(features_pkl_path):
        with open(os.path.join(DATA_DIR, config['pair_features_txt_path'])) as f:
            for line in f:
                w1, data = line.split(';')
                w2 = data.split(' ')[0]
                data = np.array(data.split(' ')[1:], dtype=np.float32)
                assert data.shape[0] == 5
                if w1 not in features:
                    features[w1] = {}
                features[w1][w2] = data
        with open(features_pkl_path, 'wb') as f:
            pkl.dump(features, f)

    with open(features_pkl_path, 'rb') as f:
        features = pkl.load(f)
        print("Sample pair features:")
        for i, w1 in enumerate(list(features.keys())):
            w2 = np.random.choice(list(features[w1].keys()))
            print(w1, w2, features[w1][w2])
            if i >= 5:
                break
    assert len(features) > 0
    return features


def build_data_and_streams(config, additional_streams=[], default_batch_size=1, seed=42):
    data_and_streams = {}
    if config["dataset"] == "snli":
        data_and_streams["data"] = SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"] == "mnli":
        data_and_streams["data"] = SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"])

    if config['useitrick']:
        features = load_pair_features(config)

    # Loading additional streams
    stream_loaders = {
        "breaking": lambda: SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "breaking")
    }
    for stream in additional_streams:
        data_and_streams["%s_data" % stream] = stream_loaders[stream]()

    def modified_stream(s):
        def _stream():
            while True:
                it = s.get_epoch_iterator()
                for x1, x1_mask, x1_lemma, x2, x2_mask, x2_lemma, y in it:
                    assert x1.shape == x1_mask.shape
                    x1 = pad_sequences(x1, maxlen=config['sentence_max_length'],
                                       padding='post', truncating='post')
                    x2 = pad_sequences(x2, maxlen=config['sentence_max_length'],
                                        padding='post', truncating='post')

                    x1_mask_padded = np.zeros(shape=(x1_mask.shape[0],
                                                     config['sentence_max_length']))
                    x2_mask_padded = np.zeros(shape=(x2_mask.shape[0],
                                                     config['sentence_max_length']))
                    x1_mask_padded[:x1_mask.shape[0], :x1_mask.shape[1]] = x1_mask
                    x2_mask_padded[:x2_mask.shape[0], :x2_mask.shape[1]] = x2_mask
                    assert x1.shape == x1_mask_padded.shape

                    model_input = [x1, x1_mask_padded, x2, x2_mask_padded]

                    if config['useitrick']:
                        kb_x, kb_y, _, _ = prepare_kb(config, features, x1_lemma, x2_lemma)
                        model_input += [kb_x, kb_y]

                    yield model_input, np_utils.to_categorical(y, 3)

        return _stream

    for stream_name in list(config["batch_sizes"].keys()) + additional_streams:
        data = data_and_streams.get("%s_data" % stream_name, data_and_streams["data"])
        stream_batch_size = config["batch_sizes"].get(stream_name, default_batch_size)
        stream = data.get_stream(stream_name, batch_size=stream_batch_size, seed=seed)
        data_and_streams[stream_name] = modified_stream(stream)()

    return data_and_streams


def compute_metrics(config, model, data_and_streams, eval_streams, default_batch_size=1):
    metrics = {}
    for stream_name in eval_streams:
        stream = data_and_streams[stream_name]
        data = data_and_streams.get("%s_data" % stream_name, data_and_streams["data"])
        num_examples = data.num_examples(stream_name)
        print(num_examples)
        print(num_examples / config["batch_sizes"].get(stream_name, default_batch_size))
        metrics[stream_name] = model.evaluate_generator(
            generator=stream,
            steps=num_examples / config["batch_sizes"].get(stream_name, default_batch_size),
            verbose=1
        )
    return metrics
