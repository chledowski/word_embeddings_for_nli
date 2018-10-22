import os

REPOSITORY_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

DATA_DIR = os.environ.get('DATA_DIR', None)

EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

LEXICONS_DIR = os.path.join(DATA_DIR, "lexicons")