"""
The `evaluate` command computes metrics of given model on all evaluation_parts of dataset.
"""


import logging

from common.experiment import Experiment
from common.paths import *
from common.utils import load_config, prepare_environment

from training.trainer import Trainer

logger = logging.getLogger(__name__)


def evaluate(model, stream):
    """
    Evaluates the model on all samples from given stream.

    :param model: ``Model`` object
    :param stream: ``NLIStream`` object to evaluate model on.
    :return: Computed metrics.
    """
    return model.evaluate_generator(
        generator=stream,
        steps=len(stream),
        verbose=True,
        use_multiprocessing=False
    )


def evaluate_from_args(args):
    """
    Evaluate model from config given in ``args.config`` attribute.
    """
    model_path = os.path.join(DATA_DIR, args.model_path)

    config = load_config(os.path.join(model_path, 'config.json'))
    rng = prepare_environment(config)

    experiment = Experiment.from_config(config, rng=rng)
    model = experiment.model

    model = Trainer.compile_model(model, config['trainer'])

    # Restore the best model found during validation
    model.load_weights(os.path.join(model_path, "best_model.h5"))

    metrics = {}
    for name in experiment.dataset.evaluation_parts:
        metrics[name] = evaluate(model, experiment.streams[name])

    print(metrics)


def evaluate_from_parser(parser):
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    evaluate_from_args(args)
