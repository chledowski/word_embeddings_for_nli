import numpy as np
import os
import pickle as pkl

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from src import DATA_DIR
from src.util.data import SNLIData
from src.util.vocab import Vocabulary
from src.models.kim.scripts.kim.data_iterator import TextIterator


def prepare_kb(config, features, vocab, x1, x2):
    batch_size = x1.shape[0]
    # print("KB batch_size: %d" % batch_size)
    kb_x = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')
    kb_y = np.zeros((batch_size, config['sentence_max_length'], config['sentence_max_length'], 5)).astype('float32')

    total_misses = 0.0
    total_hits = 0.0

    def fill_kb(features, batch_id, words1, words2, kb):
        hits, misses = 0, 0
        for i1, w1 in enumerate(words1):
            for i2, w2 in enumerate(words2):
                if w1 in features and w2 in features[w1]:
                    kb[batch_id][i1][i2] = features[w1][w2]
                    hits += 1
                else:
                    misses += 1
        return hits, misses

    sample_x1_word = None
    sample_x2_word = None
    for batch_id in range(batch_size):
        x1_words = [vocab.id_to_word(s) for s in x1[batch_id]]
        x2_words = [vocab.id_to_word(s) for s in x2[batch_id]]
        sample_x1_word = np.random.choice(x1_words)
        sample_x2_word = np.random.choice(x2_words)
        h, m = fill_kb(features, batch_id, x1_words, x2_words, kb_x)
        total_hits += h
        total_misses += m
        h, m = fill_kb(features, batch_id, x2_words, x1_words, kb_y)
        total_hits += h
        total_misses += m

    print("Hits: %d Misses: %d ratio: %.2f | x1: %s x2: %s" % (
            total_hits,
            total_misses,
            total_hits / (total_hits + total_misses),
            sample_x1_word, sample_x2_word))
    return kb_x, kb_y


def prepare_data(seqs_x, seqs_y, seqs_x_lemma, seqs_y_lemma, labels, config, kb_dict=None, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None, None

    n_samples = len(seqs_x)

    x = np.zeros((n_samples, config['sentence_max_length'])).astype('int64')
    y = np.zeros((n_samples, config['sentence_max_length'])).astype('int64')
    x_mask = np.zeros((n_samples, config['sentence_max_length'])).astype('float32')
    y_mask = np.zeros((n_samples, config['sentence_max_length'])).astype('float32')
    l = np.zeros((n_samples,)).astype('int64')
    # kb_x = np.zeros((maxlen_x, n_samples, maxlen_y, options['dim_kb'])).astype('float32')
    # kb_y = np.zeros((maxlen_y, n_samples, maxlen_x, options['dim_kb'])).astype('float32')
    # kb_att = np.zeros((maxlen_x, n_samples, maxlen_y)).astype('float32')

    for idx, [s_x, s_y, s_xl, s_yl, ll] in enumerate(zip(seqs_x, seqs_y, seqs_x_lemma, seqs_y_lemma, labels)):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1.
        y[idx, :lengths_y[idx]] = s_y
        y_mask[idx, :lengths_y[idx]] = 1.
        l[idx] = ll

        # for sid, s in enumerate(s_xl):
        #     for tid, t in enumerate(s_yl):
        #         if s in kb_dict:
        #             if t in kb_dict[s]:
        #                 kb_x[sid, idx, tid, :] = np.array(kb_dict[s][t]).astype('float32')
        #                 kb_att[sid, idx, tid] = 1.
        #
        # for sid, s in enumerate(s_yl):
        #     for tid, t in enumerate(s_xl):
        #         if s in kb_dict:
        #             if t in kb_dict[s]:
        #                 kb_y[sid, idx, tid, :] = np.array(kb_dict[s][t]).astype('float32')

    return x, x_mask, y, y_mask, l
    # return x, x_mask, kb_x, y, y_mask, kb_y, kb_att, l


class StreamWrapperKim:
    def __init__(self):
        self.force_reset = False

    def reset(self):
        self.force_reset = True

    def wrapped_stream(self, config, iterator):
        def _stream():
            while True:
                for x1, x2, x1_lemma, x2_lemma, y in iterator:
                    if self.force_reset:
                        self.force_reset = False
                        break
                    x1, x1_mask, x2, x2_mask, y = prepare_data(x1, x2, x1_lemma, x2_lemma, y, config=config)
                    model_input = [x1, x1_mask, x2, x2_mask]
                    yield model_input, np_utils.to_categorical(y, 3)
        return _stream


class StreamWrapper:
    def __init__(self):
        self.force_reset = False

    def reset(self):
        self.force_reset = True

    def modified_stream(self, config, features, data, stream=None):
        def _stream():
            while True:
                it = stream.get_epoch_iterator()
                for x1, x1_mask, x2, x2_mask, y in it:
                    if self.force_reset:
                        self.force_reset = False
                        break
                    assert x1.shape == x1_mask.shape
                    x1 = pad_sequences(x1, maxlen=config['sentence_max_length'],
                                       padding='post', truncating='post')
                    x2 = pad_sequences(x2, maxlen=config['sentence_max_length'],
                                       padding='post', truncating='post')
                    x1_mask_padded = pad_sequences(x1_mask,
                                                   maxlen=config['sentence_max_length'],
                                                   padding='post', truncating='post')
                    x2_mask_padded = pad_sequences(x2_mask,
                                                   maxlen=config['sentence_max_length'],
                                                   padding='post', truncating='post')
                    assert x1.shape == x1_mask_padded.shape
                    assert x2.shape == x2_mask_padded.shape

                    model_input = [x1, x1_mask_padded, x2, x2_mask_padded]

                    if config['useitrick']:
                        kb_x, kb_y = prepare_kb(config, features, data.vocab, x1, x2)
                        model_input += [kb_x, kb_y]

                    yield model_input, np_utils.to_categorical(y, 3)

        return _stream


def build_features(config):
    features_pkl_path = os.path.join(DATA_DIR, config['pair_features_pkl_path'])

    # Creating pickled dict file from txt.
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

    # Loading pickled dict.
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


def build_data_and_streams_like_kim(config, rng, additional_streams=[], default_batch_size=1, seed=42):
    features = None
    if config['useitrick']:
        features = build_features(config)

    class SNLIDataFromKIM:
        def __init__(self, *args, **kwargs):
            self._vocab = None
            self.iterators = dict()
            self.iterators['train'] = TextIterator(os.path.join(DATA_DIR, 'kim_data', config["datasets"][0]),
                                              os.path.join(DATA_DIR, 'kim_data', config["datasets"][1]),
                                              os.path.join(DATA_DIR, 'kim_data', config["datasets"][2]),
                                              os.path.join(DATA_DIR, 'kim_data', config["datasets"][3]),
                                              os.path.join(DATA_DIR, 'kim_data', config["datasets"][4]),
                                              os.path.join(DATA_DIR, 'kim_data', config["dictionary"][0]),
                                              os.path.join(DATA_DIR, 'kim_data', config["dictionary"][1]),
                                              n_words=config["n_words"],
                                              n_words_lemma=config["n_words_lemma"],
                                              batch_size=config["batch_sizes"]["train"],
                                              rng=rng)
            self.iterators['dev'] = TextIterator(os.path.join(DATA_DIR, 'kim_data', config["valid_datasets"][0]),
                                              os.path.join(DATA_DIR, 'kim_data', config["valid_datasets"][1]),
                                              os.path.join(DATA_DIR, 'kim_data', config["valid_datasets"][2]),
                                              os.path.join(DATA_DIR, 'kim_data', config["valid_datasets"][3]),
                                              os.path.join(DATA_DIR, 'kim_data', config["valid_datasets"][4]),
                                              os.path.join(DATA_DIR, 'kim_data', config["dictionary"][0]),
                                              os.path.join(DATA_DIR, 'kim_data', config["dictionary"][1]),
                                              n_words=config["n_words"],
                                              n_words_lemma=config["n_words_lemma"],
                                              batch_size=config["batch_sizes"]["dev"],
                                              shuffle=False,
                                              rng=rng)
            self.iterators['test'] = TextIterator(os.path.join(DATA_DIR, 'kim_data', config["test_datasets"][0]),
                                             os.path.join(DATA_DIR, 'kim_data', config["test_datasets"][1]),
                                             os.path.join(DATA_DIR, 'kim_data', config["test_datasets"][2]),
                                             os.path.join(DATA_DIR, 'kim_data', config["test_datasets"][3]),
                                             os.path.join(DATA_DIR, 'kim_data', config["test_datasets"][4]),
                                             os.path.join(DATA_DIR, 'kim_data', config["dictionary"][0]),
                                             os.path.join(DATA_DIR, 'kim_data', config["dictionary"][1]),
                                             n_words=config["n_words"],
                                             n_words_lemma=config["n_words_lemma"],
                                             batch_size=config["batch_sizes"]["test"],
                                             shuffle=False,
                                             rng=rng)

        @property
        def vocab(self):
            if not self._vocab:
                with open(os.path.join(DATA_DIR, 'kim_data', config["dictionary"][0]), 'rb') as f:
                    self._vocab = pkl.load(f)
            return self._vocab

        def num_examples(self, part):
            return {
                'train': 549367,
                'dev': 9842,
                'test': 9824
            }[part]

    data_and_streams = dict()
    data_and_streams["data"] = SNLIDataFromKIM()

    for stream_name in list(config["batch_sizes"].keys()) + additional_streams:
        data = data_and_streams.get("%s_data" % stream_name, data_and_streams["data"])
        iterator = data.iterators[stream_name]
        data_and_streams[stream_name] = StreamWrapperKim().wrapped_stream(config, iterator)()
    return data_and_streams


def build_data_and_streams(config, additional_streams=[], default_batch_size=1, seed=42):
    data_and_streams = {}
    if config["dataset"] == "snli":
        data_and_streams["data"] = SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"] == "mnli":
        data_and_streams["data"] = SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"])

    features = None
    if config['useitrick']:
        features = build_features(config)

    # Loading additional streams
    stream_loaders = {
        "breaking": lambda: SNLIData(config["train_on_fraction"], os.path.join(DATA_DIR, "snli"), "breaking")
    }
    for stream in additional_streams:
        data_and_streams["%s_data" % stream] = stream_loaders[stream]()

    for stream_name in list(config["batch_sizes"].keys()) + additional_streams:
        data = data_and_streams.get("%s_data" % stream_name, data_and_streams["data"])
        stream_batch_size = config["batch_sizes"].get(stream_name, default_batch_size)
        stream = data.get_stream(stream_name, batch_size=stream_batch_size, seed=seed)
        data_and_streams[stream_name] = StreamWrapper().modified_stream(config, features, data, stream)()

    return data_and_streams


def compute_metrics(config, model, data_and_streams, eval_streams, default_batch_size=1):
    metrics = {}
    for stream_name in eval_streams:
        stream = data_and_streams[stream_name]
        data = data_and_streams.get("%s_data" % stream_name, data_and_streams["data"])
        num_examples = data.num_examples(stream_name)
        metrics[stream_name] = model.evaluate_generator(
            generator=stream,
            steps=num_examples / config["batch_sizes"].get(stream_name, default_batch_size),
            verbose=1
        )
    return metrics
