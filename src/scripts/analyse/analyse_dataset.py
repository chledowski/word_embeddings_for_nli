import argparse
import deepdish as dd
import json
import os

from src.models import build_model
from src.util import modified_stream, evaluate_wv, load_embedding_from_h5
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics

from numpy.random import seed
from numpy.random import RandomState
import numpy as np
from tensorflow import set_random_seed





def eval_on_dataset(model, dset = 'easy', confidency = '0.45', dataset = None):
    if dataset is None:
        if dset == 'easy':
            dataset = dd.io.load('results/easy_dataset_%s.json' % confidency)
        elif dset == 'hard':
            dataset = dd.io.load('results/hard_dataset.json')
        else:
            print("Oops!  That was no valid number.  Try again...")
            return 0

    # print (dset)

    bs = len(dataset['input_premise'])
    # print(bs)
    input = [dataset['input_premise'], dataset['input_premise_mask'], dataset['input_hypothesis'], dataset['input_hypothesis_mask']]
    labels = dataset['label']
    preds = []
    # for i in range(32):
    #     logits = model.predict(input[i*(bs//32) : (i+1)*(bs//32)], batch_size=bs//32)
    #     correct_preds = np.equal(np.argmax(labels, 1), np.argmax(logits, 1))
    #     print(len(correct_preds))
    #     preds.extend(correct_preds)
    #
    # print (len(preds))
    # # print( preds)
    # print(np.mean(preds))
    # e_x = np.exp(logits)
    # smx = e_x / e_x.sum(axis=1).reshape((-1,1))
    # print(smx)
    logits = model.predict(input, batch_size=bs)
    correct_preds = np.equal(np.argmax(labels, 1), np.argmax(logits, 1))
    # _correct_preds = np.multiply(_correct_preds, np.max(smx, 1) > args.confidency )
    #     correct_preds = np.multiply(_correct_preds, correct_preds).astype(int)
    # print(np.mean(correct_preds))
    return np.mean(correct_preds)

if __name__ == "__main__":

    with open(os.path.join('results', 'stupid_snli_1', 'config.json'), 'r') as f:
        config = json.load(f)
    config["seed"] = 1
    print(config["seed"])
    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets_to_load = ["snli"]
    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=datasets_to_load)
    model = build_model(config, datasets[config["dataset"]])
    model.load_weights(os.path.join('results', 'stupid_snli_1', "best_model.h5"))

    eval_on_dataset(model, dset='easy')
    eval_on_dataset(model, dset='hard')

    # if args.produce_dset_batches:
    #     # produce_dset_batches(config, model, datasets, streams)
    #
    #     d = {name : {} for name in args.models_to_analyse}
    #     print(d)
    #
    #
    #     batch = 0
    #     for x in streams["snli"]["test"]:
    #
    #         streams[batch] = x
    #         easy_dataset = {'input_premise': [], 'input_premise_mask': [], 'input_hypothesis': [], 'input_hypothesis_mask': [], 'label': []}
    #         correct_preds = np.ones((307))
    #         for model_name in args.models_to_analyse:
    #
    #             model.load_weights(os.path.join('results', model_name, "best_model.h5"))
    #
    #             correct_pairs = []
    #             uncorrect_pairs = []
    #             correct_pairs_int = []
    #             uncorrect_pairs_int = []
    #
    #             input = x[0]
    #             input_premise = x[0][0]
    #             input_hypothesis = x[0][2]
    #             labels = x[1]
    #             logits = model.predict(input, batch_size=config["batch_sizes"]["snli"]["test"])
    #             e_x = np.exp(logits)
    #             smx = e_x / e_x.sum(axis=1).reshape((-1,1))
    #             print(smx)
    #             _correct_preds = np.equal(np.argmax(labels, 1), np.argmax(logits, 1))
    #             _correct_preds = np.multiply(_correct_preds, np.max(smx, 1) > args.confidency )
    #             correct_preds = np.multiply(_correct_preds, correct_preds).astype(int)
    #
    #             print(np.sum(correct_preds))
    #             # for i in range(len(correct_preds)):
    #             #     if correct_preds[i]:
    #             #         correct_pairs.append([datasets["snli"].vocab.decode(input_premise[i]), datasets["snli"].vocab.decode(input_hypothesis[i])])
    #             #         correct_pairs_int.append([input_premise[i], input_hypothesis[i]])
    #             #     else:
    #             #         uncorrect_pairs.append([datasets["snli"].vocab.decode(input_premise[i]), datasets["snli"].vocab.decode(input_hypothesis[i])])
    #             #         uncorrect_pairs_int.append([input_premise[i], input_hypothesis[i]])
    #             # print(np.mean(correct_preds))
    #             # print("CORRECT")
    #             # printlist(correct_pairs)
    #             # print("UNCORRECT")
    #             # printlist(uncorrect_pairs)
    #             # k = 0
    #             # for i in range(len(correct_pairs_int)):
    #
    #         print(np.sum(correct_preds))
    #         easy_dataset['input_premise'].extend(x[0][0][correct_preds])
    #         easy_dataset['input_premise_mask'].extend(x[0][1][correct_preds])
    #         easy_dataset['input_hypothesis'].extend(x[0][2][correct_preds])
    #         easy_dataset['input_hypothesis_mask'].extend(x[0][3][correct_preds])
    #         easy_dataset['label'].extend(x[1][correct_preds])
    #         # print(easy_dataset)
    #
    #         batch += 1
    #         if batch >= args.produce_dset_batches:
    #             break
    # # with open('results/easy_dataset_%s.json' % args.confidency, 'w') as f:
    # #     json.dump(results_dict, f)
    # #
    # #
    # # with open('results/%s/retrofitting_results.json' % model_name, 'w') as f:
    # #     json.dump(results_dict, f)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-name", type=str)
#     parser.add_argument("--json-path", type=str)
#     parser.add_argument("--models-to-analyse", type=list, nargs='+',
#                         default=['stupid_snli_1', 'stupid_snli_2', 'stupid_snli_3', 'stupid_snli_4', 'stupid_snli_5', ])
#     parser.add_argument("--embedding-name", type=str)
#     parser.add_argument("--produce-dset-batches", default=0, type=int) # 32
#     parser.add_argument("--confidency", default=0.45, type=float)
#
#     parser.add_argument("--compute-metrics", action='store_true')
#
#     args = parser.parse_args()
#     eval_model()
#




