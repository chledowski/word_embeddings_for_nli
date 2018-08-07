import argparse
import os

from pprint import pprint as pp

from main import train
from src import DATA_DIR
from src.configs.kim import baseline_configs
from src.scripts.preprocess_data.embedding_file_change_format import h5_to_txt

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    config = baseline_configs.get_root_config()

    embedding_path_txt = os.path.join(DATA_DIR, 'raw/embeddings/glove.840B.300d/glove.840B.300d.txt')

    # # KIM needs embedding as text file.
    # if not os.path.exists(embedding_path_txt):
    #     print("Converting embedding from H5 to TXT...")
    #     h5_to_txt(h5_name=args.embedding,
    #               txt_name=args.embedding,
    #               prefix="")

    config['embedding'] = embedding_path_txt
    config['model'] = args.model
    config['saveto'] = os.path.join(DATA_DIR, 'results', config['model'], 'model.npz')

    make_dirs([os.path.dirname(config['saveto'])])

    print("Training KIM...")
    train(**config)