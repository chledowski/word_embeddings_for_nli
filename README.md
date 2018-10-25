# NLI embeddings

### Quick reproduction

If you'd like to quickly reproduce our results:

0. Fill working directory paths in `env.sh.default` and run:

`source env.sh.default`

1. Download required input files & datasets:

`scripts/download.sh`

2. Choose training config from `training_configs` directory.

3. Run training, e.g.

`python3 run.py --command=train --config=training_configs/esim-snli0.01.json --savedir=results/esim-snli0.01`

4. Evaluate model

`python3 run.py --command=evaluate --model-path=results/esim-snli0.01`