import os

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from src import DATA_DIR
from src.util.data import SNLIData


def build_data_and_streams(config, additional_streams=[], default_batch_size=1):
    data_and_streams = {}
    if config["dataset"] == "snli":
        data_and_streams["data"] = SNLIData(os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"] == "mnli":
        data_and_streams["data"] = SNLIData(os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"])

    # Loading additional streams
    stream_loaders = {
        "breaking": lambda: SNLIData(os.path.join(DATA_DIR, "snli"), "breaking")
    }
    for stream in additional_streams:
        data_and_streams["%s_data" % stream] = stream_loaders[stream]()

    def modified_stream(s):
        def _stream():
            while True:
                it = s.get_epoch_iterator()
                for x1, _, x2, _, y in it:
                    yield [pad_sequences(x1, maxlen=config['sentence_max_length'],
                                         padding='post', truncating='post'),
                           pad_sequences(x2, maxlen=config['sentence_max_length'],
                                         padding='post', truncating='post')], np_utils.to_categorical(y, 3)

        return _stream

    for stream_name in list(config["batch_sizes"].keys()) + additional_streams:
        data = data_and_streams.get("%s_data" % stream_name, data_and_streams["data"])
        stream_batch_size = config["batch_sizes"].get(stream_name, default_batch_size)
        stream = data.get_stream(stream_name, batch_size=stream_batch_size)
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
