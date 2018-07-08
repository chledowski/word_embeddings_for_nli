import argparse
import json
import os
import pandas as pd
import pathlib

def main():
    result_dict = {}
    for json_path_posix in pathlib.Path('results').glob('**/%s*/retrofitting_results.json' % args.prefix):
        json_path = str(json_path_posix)
        embedding_name = os.path.basename(os.path.dirname(json_path))
        print(embedding_name)

        embedding_dict = {}
        result_dict[embedding_name] = embedding_dict

        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        for key, subdict in json_dict.items():
            for subkey, value in subdict.items():
                embedding_dict['%s_%s_%s' % (args.model_name, key, subkey)] = value

    df = pd.DataFrame(result_dict).round(3)
    print(df)
    pd.DataFrame.to_csv(df, 'results/%s_retrofitting_results.csv' % args.model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default='cbow', type=str)
    parser.add_argument("--prefix", default='', type=str)
    args = parser.parse_args()

    main()