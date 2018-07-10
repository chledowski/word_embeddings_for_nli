'''
Summary a source file using a summarization model.
'''
import argparse
import json
import pickle as pkl
import os

from src.models.kim.scripts.kim.data_iterator import TextIterator

from src import DATA_DIR
from src.configs.kim import baseline_configs
from src.util import evaluate_wv, load_embedding_from_h5
from src.models.kim.scripts.kim.main import (
    build_model, pred_probs, prepare_data, pred_acc, load_params, init_params, init_tparams)

def main():
    model_path = os.path.join(DATA_DIR, 'results', args.model_name, 'model.npz')
    config = baseline_configs.get_root_config()

    results_dict = {}

    # load model model_options
    with open('%s.pkl' % model_path, 'rb') as f:
        options = pkl.load(f)

    print(options)
    # load dictionary and invert
    with open(config["dictionary"][0], 'rb') as f:
        word_dict = pkl.load(f)

    print('Loading knowledge base ...')
    kb_dicts = options['kb_dicts']
    with open(kb_dicts[0], 'rb') as f:
        kb_dict = pkl.load(f)

    n_words = options['n_words']
    valid_batch_size = options['valid_batch_size']

    print('Preparing datasets ...')

    datasets_names = ["train", "valid", "test", "breaking"]
    config_names = ["datasets", "valid_datasets", "test_datasets", "breaking_datasets"]
    datasets = []

    for config_name in config_names:
        datasets.append(
            TextIterator(config[config_name][0],
                         config[config_name][1],
                         config[config_name][2],
                         config[config_name][3],
                         config[config_name][4],
                         config["dictionary"][0], config["dictionary"][1],
                         n_words=n_words,
                         batch_size=valid_batch_size,
                         shuffle=False)
        )



    print('Loading model...')

    # allocate model parameters
    params = init_params(options, word_dict)

    # load model parameters and set theano shared variables
    params = load_params(model_path, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x1, x1_mask, x1_kb, x2, x2_mask, x2_kb, kb_att, y, \
        opt_ret, \
        cost, \
        f_pred, \
        f_probs = \
        build_model(tparams, options)

    results_dict['accuracies'] = {}

    use_noise.set_value(0.)

    print('Computing accuracy...')

    for dataset_name, dataset in zip(datasets_names, datasets):
        acc = pred_acc(f_pred, prepare_data, options, dataset, kb_dict)
        print('%s accuracy' % dataset_name, acc)
        results_dict['accuracies'][dataset_name] = acc

    #
    # predict_labels_valid = pred_label(f_pred, prepare_data, options, valid, kb_dict)
    # predict_labels_test = pred_label(f_pred, prepare_data, options, test, kb_dict)
    #
    # with open('predict_gold_samples_valid.txt', 'w') as fw:
    #     with open(config["valid_datasets"][0], 'r') as f1:
    #         with open(config["valid_datasets"][1], 'r') as f2:
    #             with open(config["valid_datasets"][-1], 'r') as f3:
    #                 for a, b, c, d in zip(predict_labels_valid, f3, f1, f2):
    #                     fw.write(str(a) + '\t' + b.rstrip() + '\t' + c.rstrip() + '\t' + d.rstrip() + '\n')
    #
    # with open('predict_gold_samples_test.txt', 'w') as fw:
    #     with open(config["test_datasets"][0], 'r') as f1:
    #         with open(config["test_datasets"][1], 'r') as f2:
    #             with open(config["test_datasets"][-1], 'r') as f3:
    #                 for a, b, c, d in zip(predict_labels_test, f3, f1, f2):
    #                     fw.write(str(a) + '\t' + b.rstrip() + '\t' + c.rstrip() + '\t' + d.rstrip() + '\n')

    print("Evaluating embedding...")

    _, _, wv = load_embedding_from_h5(args.embedding)
    results_dict['backup'] = evaluate_wv(wv, simlex_only=False)

    with open('results/%s/retrofitting_results.json' % args.model_name, 'w') as f:
        json.dump(results_dict, f)

    print('Done')

def pred_label(f_pred, prepare_data, options, iterator, kb_dict):
    labels = []
    for x1, x2, x1_lemma, x2_lemma, y in iterator:
        x1, x1_mask, x1_kb, x2, x2_mask, x2_kb, kb_att, y = prepare_data(x1, x2, x1_lemma, x2_lemma, y, options, kb_dict)
        preds = f_pred(x1, x1_mask, x1_kb, x2, x2_mask, x2_kb, kb_att)
        labels = labels + preds.tolist()

    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--embedding", required=True, type=str)
    args = parser.parse_args()
    main()
