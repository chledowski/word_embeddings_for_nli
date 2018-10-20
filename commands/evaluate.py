#!/usr/bin/env python

import argparse
import copy
import logging
import os

from common.paths import *
from common.utils import load_config, prepare_environment
from data.dataset import NLIData
from data.embedding import NLIEmbedding
from data.stream import NLIStream
from data.transformers import NLITransformer
from data.vocabulary import NLIVocabulary

from src.models.esim import esim

logger = logging.getLogger(__name__)


def evaluate_from_args(args):
    config = load_config(args.model_path)
    rng = prepare_environment(config)

    # 0. Load dataset
    dataset = NLIData.from_config(config['dataset'])

    # 1. Load vocabularies
    vocabs = {}
    for name, vocab_config in config['vocabs'].items():
        vocab_config['file_or_data'] = os.path.join(dataset.path, vocab_config['file_or_data'])
        vocabs[name] = NLIVocabulary.from_config(config=vocab_config)

    # 2. Load embeddings
    embeddings = {}
    for name, emb_config in config['embeddings'].items():
        emb_config['file'] = os.path.join(EMBEDDINGS_DIR, emb_config['file'])
        embeddings[name] = NLIEmbedding.from_config(config=emb_config,
                                                    rng=rng,
                                                    vocabs=vocabs)

    # 3. Batch transformers
    batch_transformers = []
    for bt_config in config['batch_transformers']:
        bt_config = copy.deepcopy(bt_config)
        if 'vocab' in bt_config:
            bt_config['vocab'] = vocabs.get(bt_config['vocab'])
        transformer = NLITransformer.by_name(bt_config['name']).from_config(bt_config)
        batch_transformers.append(transformer)

    # 4. Load streams
    streams = {}
    for name in dataset.parts:
        streams[name] = NLIStream.from_config(
            config=config['streams'][name],
            dataset=dataset.part(name),
            rng=rng,
            batch_transformers=batch_transformers)

    model = esim(config=config['model'],
                 embeddings=embeddings)

    # Restore the best model found during validation
    model.load_weights(os.path.join(args.model_path, "best_model.h5"))

    metrics = {}
    for name in dataset.evaluation_parts:
        metrics[name] = model.evaluate_generator(
            generator=streams[name],
            steps=len(streams[name]),
            verbose=True,
            use_multiprocessing=False
        )

    print(metrics)

    # results_dict = {}
    # results_dict['accuracies'] = {}
    # results_dict['losses'] = {}
    # for stream_name, stream_metrics in metrics.items():
    #     loss, accuracy = stream_metrics
    #     print('{} loss / accuracy = {:.4f} / {:4f}'.format(stream_name, loss, accuracy))
    #     results_dict['accuracies'][stream_name] = accuracy
    #     results_dict['losses'][stream_name] = loss
    #
    # model_name = args.model_name.split('/')[0]
    #
    # with open('results/%s/retrofitting_results.json' % model_name, 'w') as f:
    #     json.dump(results_dict, f)


def evaluate(parser):
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    evaluate_from_args(args)
