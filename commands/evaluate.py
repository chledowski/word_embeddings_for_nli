#!/usr/bin/env python

import logging

from common.experiment import Experiment
from common.paths import *
from common.utils import load_config, prepare_environment

from training.trainer import Trainer

logger = logging.getLogger(__name__)


def evaluate(model, stream):
    return model.evaluate_generator(
        generator=stream,
        steps=len(stream),
        verbose=True,
        use_multiprocessing=False
    )


def evaluate_from_args(args):
    config = load_config(os.path.join(args.model_path, 'config.json'))
    rng = prepare_environment(config)

    experiment = Experiment.from_config(config, rng=rng)
    model = experiment.model

    model = Trainer.compile_model(model, config['trainer'])

    # Restore the best model found during validation
    model.load_weights(os.path.join(args.model_path, "best_model.h5"))

    metrics = {}
    for name in experiment.dataset.evaluation_parts:
        metrics[name] = evaluate(model, experiment.streams[name])

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


def evaluate_from_parser(parser):
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    evaluate_from_args(args)
