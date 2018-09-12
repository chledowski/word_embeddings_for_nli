#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import os
import deepdish as dd

from src.models import build_model
from src.util import modified_stream, evaluate_wv, load_embedding_from_h5
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics
from src.scripts.analyse.analyse_dataset import eval_on_dataset

from numpy.random import seed
from numpy.random import RandomState
import numpy as np
from tensorflow import set_random_seed


def eval_model():

    with open(os.path.join('results', args.model_name, 'config.json'), 'r') as f:
        config = json.load(f)
    print(config["seed"])
    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets_to_load = ["snli"]
    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=datasets_to_load)
    model = build_model(config, datasets[config["dataset"]])

    easy_dataset = {'input_premise': [], 'input_premise_mask': [], 'input_hypothesis': [], 'input_hypothesis_mask': [], 'label': []}
    hard_dataset = {'input_premise': [], 'input_premise_mask': [], 'input_hypothesis': [], 'input_hypothesis_mask': [], 'label': []}
    batch = 0

    for x in streams["snli"]["test"]:

        streams[batch] = x
        correct_preds = np.ones((307))
        incorrect_preds = np.ones((307))
        for model_name in args.models_to_analyse:

            model.load_weights(os.path.join('results', model_name, "best_model.h5"))

            correct_pairs = []
            incorrect_pairs = []
            correct_pairs_int = []
            incorrect_pairs_int = []

            input = x[0]
            input_premise = x[0][0]
            input_hypothesis = x[0][2]
            labels = x[1]
            logits = model.predict(input, batch_size=config["batch_sizes"]["snli"]["test"])
            e_x = np.exp(logits)
            smx = e_x / e_x.sum(axis=1).reshape((-1,1))

            _correct_preds = np.equal(np.argmax(labels, 1), np.argmax(logits, 1))
            _correct_preds = np.multiply(_correct_preds, np.max(smx, 1) > args.confidency )
            correct_preds = np.multiply(_correct_preds, correct_preds).astype(int)


            _incorrect_preds = np.not_equal(np.argmax(labels, 1), np.argmax(logits, 1))
            incorrect_preds = np.multiply(_incorrect_preds, incorrect_preds).astype(int)
            print("partial corr/incorr: {}/ {}".format(np.mean(correct_preds), np.mean(incorrect_preds)))
            # for i in range(len(correct_preds)):
            #     if correct_preds[i]:
            #         correct_pairs.append([datasets["snli"].vocab.decode(input_premise[i]), datasets["snli"].vocab.decode(input_hypothesis[i])])
            #         correct_pairs_int.append([input_premise[i], input_hypothesis[i]])
            #     else:
            #         incorrect_pairs.append([datasets["snli"].vocab.decode(input_premise[i]), datasets["snli"].vocab.decode(input_hypothesis[i])])
            #         incorrect_pairs_int.append([input_premise[i], input_hypothesis[i]])
            # print(np.mean(correct_preds))
            # print("CORRECT")
            # printlist(correct_pairs)
            # print("UNCORRECT")
            # printlist(incorrect_pairs)
            # k = 0
            # for i in range(len(correct_pairs_int)):

        print("correct preds: ", np.mean(correct_preds))
        print("incorrect preds: ", np.mean(incorrect_preds))
        correct_preds = correct_preds.astype(bool)
        incorrect_preds = incorrect_preds.astype(bool)
        easy_dataset['input_premise'].extend(x[0][0][correct_preds])
        easy_dataset['input_premise_mask'].extend(x[0][1][correct_preds])
        easy_dataset['input_hypothesis'].extend(x[0][2][correct_preds])
        easy_dataset['input_hypothesis_mask'].extend(x[0][3][correct_preds])
        easy_dataset['label'].extend(x[1][correct_preds])
        hard_dataset['input_premise'].extend(x[0][0][incorrect_preds])
        hard_dataset['input_premise_mask'].extend(x[0][1][incorrect_preds])
        hard_dataset['input_hypothesis'].extend(x[0][2][incorrect_preds])
        hard_dataset['input_hypothesis_mask'].extend(x[0][3][incorrect_preds])
        hard_dataset['label'].extend(x[1][incorrect_preds])
        # print(easy_dataset)

        batch += 1
        if batch >= args.produce_dset_batches:
            break


    dd.io.save('results/hard_dataset.json', hard_dataset)
    dd.io.save('results/easy_dataset_%s.json' % args.confidency, easy_dataset)

    easy_dataset = dd.io.load('results/easy_dataset_0.45.json')
    hard_dataset = dd.io.load('results/hard_dataset.json')

    for model_name in args.models_to_analyse:
        model.load_weights(os.path.join('results', model_name, "best_model.h5"))
        print("eval on {}".format(model_name))
        eval_on_dataset(model, dset='easy', dataset= easy_dataset)
        eval_on_dataset(model, dset='hard', dataset= hard_dataset)



    # dd.io.save('results/hard_dataset.json', hard_dataset)
    # dd.io.save('results/easy_dataset_%s.json' % args.confidency, easy_dataset)

    with open(os.path.join('results', 'stupid_snli_1', 'config.json'), 'r') as f:
        config = json.load(f)
    # config["seed"] = 1
    # print(config["seed"])
    # seed(config["seed"])
    # set_random_seed(config["seed"])
    # rng = RandomState(config["seed"])

    datasets_to_load = ["snli"]
    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=datasets_to_load)
    model = build_model(config, datasets[config["dataset"]])

    eval_on_dataset(model, dset='easy')
    eval_on_dataset(model, dset='hard')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='stupid_snli_1')
    parser.add_argument("--models-to-analyse", type=list, nargs='+',
                        default=['stupid_snli_1', 'stupid_snli_2', 'stupid_snli_3', 'stupid_snli_4', 'stupid_snli_5', ])
    parser.add_argument("--embedding-name", type=str)
    parser.add_argument("--produce-dset-batches", default=32, type=int) # 32
    parser.add_argument("--confidency", default=0.45, type=float)

    parser.add_argument("--compute-metrics", action='store_true')
    # parser.add_argument("--easy-dataset", action='store_true')
    # parser.add_argument("--hard-dataset", action='store_true')

    args = parser.parse_args()
    eval_model()
