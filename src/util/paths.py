import pathlib

from src import DATA_DIR

PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()
SRC_ROOT = PROJECT_ROOT / "src"

DATA_DIR = pathlib.Path(DATA_DIR).resolve()
EMBEDDINGS_DIR = pathlib.Path(DATA_DIR) / "embeddings"


def get_embedding_path(embedding_name):
    if embedding_name != "random_uniform":
        return EMBEDDINGS_DIR / ("%s.h5" % embedding_name)
    else:
        return None
