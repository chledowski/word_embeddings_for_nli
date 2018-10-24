"""
The helper `dump` command dumps all intermediate layer outputs of given model.
"""

import logging

from common.experiment import Experiment
from common.utils import load_config, prepare_environment, get_all_layer_outputs, save_outputs

logger = logging.getLogger(__name__)


def dump_from_args(args):
    config = load_config(args.config)
    rng = prepare_environment(config)

    experiment = Experiment.from_config(config, rng=rng)
    model = experiment.model

    names, outputs = get_all_layer_outputs(model, experiment.streams.train)
    save_outputs(args.savedir, names, outputs)


def dump_from_parser(parser):
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--savedir", type=str, default='dumps/')
    args = parser.parse_args()

    dump_from_args(args)
