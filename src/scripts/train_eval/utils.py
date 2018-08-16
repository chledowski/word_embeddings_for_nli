import numpy as np
import os
import pickle as pkl

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from src import DATA_DIR
from src.util.data import NLIData
from src.util.vocab import Vocabulary

import tensorflow as tf


def prepare_kb(config, features, x1_lemma, x2_lemma):
    batch_size = x1_lemma.shape[0]
    # print("KB batch_size: %d" % batch_size)
    kb_x = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')
    kb_y = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')

    total_misses = 0
    total_hits = 0
    pairs = []

    def fill_kb(batch_id, words1, words2, kb):
        hits, misses = 0, 0
        for i1, w1 in enumerate(words1):
            for i2, w2 in enumerate(words2):
                if type(w1) is bytes:
                    w1 = w1.decode()
                if type(w2) is bytes:
                    w2 = w2.decode()
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
        print("Sample pair features:")
        for i, w1 in enumerate(list(features.keys())):
            w2 = np.random.choice(list(features[w1].keys()))
            print(w1, w2, features[w1][w2])
            if i >= 5:
                break
    assert len(features) > 0
    return features


def build_data_and_streams(config, rng, datasets_to_load=[], default_batch_size=1):
    datasets_loaders = {
        "snli": lambda: NLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "snli"),
        "mnli": lambda: NLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "mnli"), "mnli"),
        "breaking": lambda: NLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "breaking"), "breaking",
                                    vocab_dir=os.path.join(DATA_DIR, "snli"))
    }

    datasets = {}
    for dataset_name in datasets_to_load:
        datasets[dataset_name] = datasets_loaders[dataset_name]()

    if config['useitrick']:
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
                        if len(batch) == 5:
                            x1, x1_mask, x2, x2_mask, y = batch
                        elif len(batch) == 7:
                            x1, x1_mask, x1_lemma, x2, x2_mask, x2_lemma, y = batch
                        else:
                            raise ValueError("Expected 5 or 7 elements in batch. Got %d" % len(batch))

                        x1 = pad_sequences(x1, maxlen=config['sentence_max_length'],
                                           padding='post', truncating='post')
                        x2 = pad_sequences(x2, maxlen=config['sentence_max_length'],
                                            padding='post', truncating='post')

                        x1_mask_padded = pad_sequences(x1_mask, maxlen=config['sentence_max_length'],
                                           padding='post', truncating='post')
                        x2_mask_padded = pad_sequences(x2_mask, maxlen=config['sentence_max_length'],
                                           padding='post', truncating='post')
                        assert x1.shape == x1_mask_padded.shape

                        model_input = [x1, x1_mask_padded, x2, x2_mask_padded]

                        if config['useitrick']:
                            kb_x, kb_y, _, _ = prepare_kb(config, features, x1_lemma, x2_lemma)
                            model_input += [kb_x, kb_y]

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
            metrics["%s_%s" % (dataset_name, stream_name)] = model.evaluate_generator(
                generator=stream,
                steps=num_examples / stream_batch_size,
                verbose=1
            )
    return metrics
