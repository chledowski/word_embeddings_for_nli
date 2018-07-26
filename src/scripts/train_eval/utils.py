import numpy as np
import os
import pickle as pkl

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from src import DATA_DIR
from src.util.data import SNLIData
from src.util.vocab import Vocabulary

import tensorflow as tf


def build_data_and_streams(config, additional_streams=[], default_batch_size=1, seed=42):
    data_and_streams = {}
    if config["dataset"] == "snli":
        data_and_streams["data"] = SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"] == "mnli":
        data_and_streams["data"] = SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"])

    vocabulary = Vocabulary(
        os.path.join(DATA_DIR, config["dataset"], 'vocab.txt')
    )

    features = {}

    features_pkl_path = os.path.join(DATA_DIR, config['pair_features_pkl_path'])
    if not os.path.exists(features_pkl_path):
        with open(os.path.join(DATA_DIR, config['pair_features_txt_path'])) as f:
            shown = False
            for line in f:
                w1, data = line.split(';')
                w2 = data.split(' ')[0]
                data = np.array(data.split(' ')[1:], dtype=np.float32)
                assert data.shape[0] == 5
                if w1 not in features:
                    features[w1] = {}
                features[w1][w2] = data
                # DEBUG
                if not shown:
                    print("Sample pair features:")
                    print(w1, w2, features[w1][w2])
                    shown = True
        with open(features_pkl_path, 'wb') as f:
            pkl.dump(features, f)

    with open(features_pkl_path, 'rb') as f:
        features = pkl.load(f)
    assert len(features) > 0

    # Loading additional streams
    stream_loaders = {
        "breaking": lambda: SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "breaking")
    }
    for stream in additional_streams:
        data_and_streams["%s_data" % stream] = stream_loaders[stream]()

    def prepare_kb(x1, x2):
        batch_size = x1.shape[0]
        # print("KB batch_size: %d" % batch_size)
        kb_x = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')
        kb_y = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')

        # print("x1 shape", x1.shape)
        # print("kb_x shape", kb_x.shape)

        def fill_kb(batch_id, words1, words2, kb):
            for i1, w1 in enumerate(words1):
                for i2, w2 in enumerate(words2):
                    if w1 in features and w2 in features[w1]:
                        kb[batch_id][i1][i2] = features[w1][w2]

        for batch_id in range(batch_size):
            x1_words = [vocabulary.word_to_id(s) for s in x1[batch_id]]
            x2_words = [vocabulary.word_to_id(s) for s in x2[batch_id]]
            fill_kb(batch_id, x1_words, x2_words, kb_x)
            fill_kb(batch_id, x2_words, x1_words, kb_y)
        return kb_x, kb_y

    def modified_stream(s):
        def _stream():
            while True:
                it = s.get_epoch_iterator()
                for x1, x1_mask, x2, x2_mask, y in it:
                    # decode
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

                    kb_x, kb_y = prepare_kb(x1, x2)
                    yield [x1, x1_mask_padded, x2, x2_mask_padded, kb_x, kb_y], np_utils.to_categorical(y, 3)

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
        metrics[stream_name] = model.evaluate_generator(
            generator=stream,
            steps=num_examples / config["batch_sizes"].get(stream_name, default_batch_size)
        )
    return metrics

