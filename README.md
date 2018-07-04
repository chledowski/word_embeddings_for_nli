#word_embeddings_for_nli

Before doing anything, set your DATA_DIR.

### Downloading SNLI and pretrained word embeddings
In order to download and preproccess SNLI dataset, run `bash src/scripts/fetch_data/fetch_and_preprocess_snli.sh`

To download pretrained word embeddings run `python src/scripts/fetch_data/fetch_embeddings.py`. You can choose to download only Glove by using argument `--embedding glove` or only Glove trained on wikipedia by using `--embedding wiki`.

To train a model (currently supported are: cbow, bilstm, esim) run (for cbow with glove wiki): `python src/scripts/train_eval/train_cbow.py root results/yourdir --embedding_name=wiki`



Link to related work and early results:
https://drive.google.com/open?id=1mReW69XrHbMR7rqlp1_fepLk5bFcmjhU
