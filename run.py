import argparse
import logging
import os

if os.environ.get("NLI_DEBUG") == 1:
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(level=LEVEL,
                    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from commands.evaluate import evaluate_from_parser
from commands.train import train_from_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True)
    args, _ = parser.parse_known_args()

    if args.command == 'evaluate':
        run_fn = evaluate_from_parser
    elif args.command == 'train':
        run_fn = train_from_parser
    else:
        raise ValueError("Unknown command: %s" % args.command)

    run_fn(parser)
