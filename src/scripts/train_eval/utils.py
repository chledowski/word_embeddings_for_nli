import logging
import numpy as np
import os
import pickle as pkl

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from src import DATA_DIR
from src.util.data import NLIData
from src.util.vocab import Vocabulary

import tensorflow as tf

logger = logging.getLogger(__name__)


def prepare_kb(config, features, x1_lemma, x2_lemma, x1_length, x2_length):
    batch_size = x1_lemma.shape[0]
    # print("KB batch_size: %d" % batch_size)
    kb_x = np.zeros((batch_size, x1_length, x2_length, 5)).astype('float32')
    kb_y = np.zeros((batch_size, x2_length, x1_length, 5)).astype('float32')

    total_misses = 0
    total_hits = 0
    pairs = []

    def fill_kb(batch_id, words1, words2, kb):
        hits, misses = 0, 0
        for i1 in range(len(words1)):
            w1 = words1[i1].decode()
            for i2 in range(len(words2)):
                w2 = words2[i2].decode()
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

    # sample_pair = np.random.choice(pairs)
    # print("Hits: %d Misses: %d Size: %d" % (total_hits, total_misses, len(features)))
    # print("Sample pair: %s" % sample_pair)
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
        # print("Sample pair features:")
        for i, w1 in enumerate(list(features.keys())):
            w2 = np.random.choice(list(features[w1].keys()))
            # print(w1, w2, features[w1][w2])
            if i >= 5:
                break
    assert len(features) > 0
    return features


def build_data_and_streams(config, rng, datasets_to_load=[], default_batch_size=1):
    datasets_loaders = {
        "snli": lambda: NLIData(config, os.path.join(DATA_DIR, "snli"), "snli"),
        "mnli": lambda: NLIData(config, os.path.join(DATA_DIR, "mnli"), "mnli"),
        "breaking": lambda: NLIData(config, os.path.join(DATA_DIR, "breaking"), "breaking",
                                    vocab_dir=os.path.join(DATA_DIR, "snli"))
    }

    datasets = {}
    for dataset_name in datasets_to_load:
        datasets[dataset_name] = datasets_loaders[dataset_name]()

    if config['useitrick'] or config['useatrick'] or config['usectrick'] or config['fullkim']:
        features = load_pair_features(config)

    class StreamWrapper:
        def __init__(self):
            self.force_reset = False

        def reset(self):
            self.force_reset = True

        def wrapped_stream(self, stream):
            def _stream():
                while True:
                    it = stream.get_epoch_iterator()
                    for batch in it:
                        if self.force_reset:
                            self.force_reset = False
                            break
                        if len(batch) not in [5, 7, 9]:
                            raise ValueError("Expected 5 or 7 or 9 elements in batch. Got %d" % len(batch))

                        use_external_knowledge = (
                                config['useitrick'] or
                                config['useatrick'] or
                                config['usectrick'] or
                                config['fullkim'])

                        if config['use_elmo']:
                            if use_external_knowledge:
                                x1, x1_mask, x1_lemma, x2, x2_mask, x2_lemma, y, x1_elmo, x2_elmo = batch
                            else:
                                x1, x1_mask, x2, x2_mask, y, x1_elmo, x2_elmo = batch
                        else:
                            if use_external_knowledge:
                                x1, x1_mask, x1_lemma, x2, x2_mask, x2_lemma, y = batch
                            else:
                                x1, x1_mask, x2, x2_mask, y = batch

                        def _pad(x, length):
                            return pad_sequences(x, maxlen=length,
                                                  padding='post', truncating='post')

                        x1_length = np.max(np.sum(x1_mask, axis=1)).astype(np.int32)
                        x2_length = np.max(np.sum(x2_mask, axis=1)).astype(np.int32)
                        x1 = _pad(x1, x1_length)
                        x2 = _pad(x2, x2_length)
                        x1_mask = _pad(x1_mask, x1_length)
                        x2_mask = _pad(x2_mask, x2_length)

                        if config['use_elmo']:
                            x1_elmo = _pad(x1_elmo, x1_length)
                            x2_elmo = _pad(x2_elmo, x2_length)

                        model_input = [x1, x1_mask, x2, x2_mask]

                        if use_external_knowledge:
                            kb_x, kb_y, _, _ = prepare_kb(
                                config, features, x1_lemma, x2_lemma, x1_length, x2_length)
                            model_input += [kb_x, kb_y]

                        if config['use_elmo']:
                            model_input += [x1_elmo, x2_elmo]

                        yield model_input, np_utils.to_categorical(y, 3)

            return _stream

    streams = {}
    for dataset_name, dataset in datasets.items():
        for part_name in dataset.part_map.keys():
            dataset_batch_sizes = config["batch_sizes"].get(dataset_name, {})
            stream_batch_size = dataset_batch_sizes.get(part_name, default_batch_size)
            should_shuffle = config["shuffle"].get(dataset_name, {}).get(part_name, False)
            stream = dataset.get_stream(part_name,
                                        shuffle=should_shuffle,
                                        rng=rng,
                                        batch_size=stream_batch_size)
            if dataset_name not in streams:
                streams[dataset_name] = {}
            streams[dataset_name][part_name] = StreamWrapper().wrapped_stream(stream)()
    return datasets, streams


def compute_metrics(config, model, datasets, streams, eval_streams, default_batch_size=1):
    metrics = {}
    for dataset_name, dataset in datasets.items():
        for stream_name in eval_streams:
            if stream_name not in streams[dataset_name]:
                continue
            stream = streams[dataset_name][stream_name]
            num_examples = dataset.num_examples(stream_name)
            dataset_batch_sizes = config["batch_sizes"].get(dataset_name, {})
            stream_batch_size = dataset_batch_sizes.get(stream_name, default_batch_size)

            if num_examples % stream_batch_size > 0:
                logger.warning("num_examples %d is not divisible by batch_size %d!" % (
                    num_examples, stream_batch_size
                ))

            metrics["%s_%s" % (dataset_name, stream_name)] = model.evaluate_generator(
                generator=stream,
                steps=num_examples / stream_batch_size,
                verbose=1,
                use_multiprocessing=False
            )

    return metrics

# def display_instances(config, model, datasets, streams):
#     good = []
#     bad = []
#
#
#     print(results)