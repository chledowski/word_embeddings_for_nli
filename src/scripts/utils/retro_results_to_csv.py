import json
import pandas as pd

def to_csv():
    with open('results/retrofitting_results.json', 'r') as f:
        results_dict = json.load(f)

    d = {}
    for key in results_dict.keys():
        d[key] = {}
        d[key]['lexicon'] = results_dict[key]['lexicon']
        if 'evaluation' in results_dict[key].keys():
            for key2 in results_dict[key]['evaluation'].keys():
                d[key][key2] = results_dict[key]['evaluation'][key2]
        if 'esim_acc' in results_dict[key].keys():
            for key2 in results_dict[key]['esim_acc'].keys():
                d[key]["esim_a_" + key2] = results_dict[key]['esim_acc'][key2]
        if 'esim_loss' in results_dict[key].keys():
            for key2 in results_dict[key]['esim_loss'].keys():
                d[key]["esim_l_" + key2] = results_dict[key]['esim_loss'][key2]
        if 'backup' in results_dict[key].keys():
            for key2 in results_dict[key]['backup'].keys():
                d[key]['bckp_' + key2] = results_dict[key]['backup'][key2]

    df = pd.DataFrame(d)
    df = df.round(3)
    print(df)
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