import json
import os
import pandas as pd
import pathlib

def to_csv():
    result_dict = {}
    for json_path_posix in pathlib.Path('results').glob('**/*retrofitting_results.json'):
        json_path = str(json_path_posix)
        model_name = os.path.basename(os.path.dirname(json_path))
        print(model_name)

        model_dict = {}
        result_dict[model_name] = model_dict

        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        for key in json_dict.keys():
            model_dict[key] = {}
            if 'evaluation' in json_dict[key].keys():
                for key2 in json_dict[key]['evaluation'].keys():
                    model_dict[key][key2] = json_dict[key]['evaluation'][key2]
            if 'esim_acc' in json_dict[key].keys():
                for key2 in json_dict[key]['esim_acc'].keys():
                    model_dict[key]["esim_a_" + key2] = json_dict[key]['esim_acc'][key2]
            if 'esim_loss' in json_dict[key].keys():
                for key2 in json_dict[key]['esim_loss'].keys():
                    resumodel_dictlt_dict[key]["esim_l_" + key2] = json_dict[key]['esim_loss'][key2]
            if 'backup' in json_dict[key].keys():
                for key2 in json_dict[key]['backup'].keys():
                    model_dict[key]['bckp_' + key2] = json_dict[key]['backup'][key2]

    df = pd.DataFrame(result_dict)
    df = df.round(3)
    pd.DataFrame.to_csv(df, "results/retrofitting_results.csv")

def rm_emb_from_dict(emb_name):
    with open('results/retrofitting_results.json', 'r') as f:
        results_dict = json.load(f)

    del results_dict[emb_name]

    with open('results/retrofitting_results.json', 'w') as f:
        json.dump(results_dict, f)


if __name__ == '__main__':
    to_csv()
    # rm_emb_from_dict('wiki_fq_12_q')
    # rm_emb_from_dict('wiki_fq_2_q')
    # rm_emb_from_dict('wiki_fq_12')
    # rm_emb_from_dict('wiki_fq_2')
    # rm_emb_from_dict('fq_12')
    # rm_emb_from_dict('fq_2')