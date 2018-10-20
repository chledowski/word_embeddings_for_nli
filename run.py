import argparse
import logging
import os

if os.environ.get("NLI_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(level=LEVEL,
                    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from commands.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args, _ = parser.parse_known_args()
    print(args)
    args, _ = parser.parse_known_args()
    print(args)

    if args.task == 'evaluate':
        run_fn = evaluate
    elif args.task == 'train':
        raise NotImplementedError()
    else:
        raise ValueError()

    run_fn(parser)
