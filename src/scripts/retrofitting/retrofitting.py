import argparse
import json
import random
from scipy.stats import ortho_group
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import CosineSimilarity

from web.analogy import *
from src.util import *

cos = CosineSimilarity(dim=1, eps=1e-6)


def faruqui(wv_original, lexicon, n_epochs, losses, losses_2):

    if args.optimization == 'eiu':

        if 1 in args.losses:
            a = args.alpha
        else:
            a = 0

        retro_wv = deepcopy(wv_original)
        vocab_wv = set(retro_wv.keys())
        vocab_intersection = vocab_wv.intersection(set(lexicon.keys()))

        for it in range(n_epochs):
            # loop through every node also in ontology (else just use data estimate)
            for word in vocab_intersection:
                word_neighbours = set(lexicon[word]).intersection(vocab_wv)
                n_neighbours = len(word_neighbours)
                # no neighbours, pass - use data estimate
                if n_neighbours == 0:
                    continue
                # the weight of the data estimate if the number of neighbours
                newVec = a * n_neighbours * wv_original[word]
                # loop over neighbours and add to new vector (currently with weight 1)
                for word_neighbour in word_neighbours:
                    newVec += args.beta * retro_wv[word_neighbour]
                retro_wv[word] = newVec / ((1 + args.beta) * n_neighbours)

        return retro_wv, []

    else:
        vocab_wv = set(wv_original.keys())
        vocab_lexicon = set(lexicon.keys())
        vocab_intersection = vocab_wv.intersection(vocab_lexicon)
        wv_emb_matrix, word_to_id, id_to_word = dict_to_matrix_and_id(wv_original)
        wv_emb_matrix = torch.from_numpy(wv_emb_matrix)
        retro_emb_matrix = copy.deepcopy(wv_emb_matrix)

        train_set = []

        for word in vocab_intersection:

            word_neighbours = set(lexicon[word]).intersection(vocab_wv)
            n_neighbours = len(word_neighbours)
            if n_neighbours == 0:
                continue

            for word_neighbour in word_neighbours:
                train_set.append([word, word_neighbour, 1. / n_neighbours])

        print("train_set prepared! Training...")
        loss_1_list, loss_2_list, loss_3_list = [[], [], []]
        no_batches = len(train_set) // args.batch_size

        for epoch in tqdm(range(n_epochs)):

            if 2 * epoch >= n_epochs and losses != losses_2:
                wv_retro = {}
                for key in word_to_id.keys():
                    wv_retro[key] = retro_emb_matrix[word_to_id[key]].numpy()
                # evaluate_wv(wv_retro, [])
                print("changing losses from {} to {}.".format(losses, losses_2))
                losses = losses_2

            random.shuffle(train_set)

            for batch in range(no_batches):
                train_batch = train_set[batch * args.batch_size: (batch + 1) * args.batch_size]
                word_list_batch = [train_batch[j][0] for j in range(len(train_batch))]
                word_neighbour_list_batch = [train_batch[j][1] for j in range(len(train_batch))]
                beta_batch = torch.tensor([train_batch[j][2] for j in range(len(train_batch))]).reshape(-1, 1)
                other_batch = random.sample(vocab_wv, args.batch_size)

                id_word_batch = [word_to_id[word] for word in word_list_batch]
                id_neighbour_batch = [word_to_id[word] for word in word_neighbour_list_batch]
                id_other_batch = [word_to_id[word] for word in other_batch]

                retro_word = Variable(copy.deepcopy(retro_emb_matrix[id_word_batch]), requires_grad=True)

                if args.train_var == "all":
                    retro_other = Variable(retro_emb_matrix[id_other_batch], requires_grad=True)
                    retro_neighbour = Variable(retro_emb_matrix[id_neighbour_batch], requires_grad=True)
                else:
                    retro_other = Variable(retro_emb_matrix[id_other_batch])
                    retro_neighbour = Variable(retro_emb_matrix[id_neighbour_batch])

                word = Variable(wv_emb_matrix[id_word_batch])
                other = Variable(wv_emb_matrix[id_other_batch])

                if args.optimization == "sgd":
                    opt = torch.optim.SGD([retro_word], lr=10/1.5**epoch)
                elif args.optimization == "adam":
                    opt = torch.optim.Adam([retro_word])
                else:
                    raise ValueError('Not implemented optimizer choosen.')
                opt.zero_grad()

                loss_1 = args.alpha * torch.sum((retro_word - word) ** 2) / args.batch_size
                loss_2 = args.beta * torch.sum(beta_batch * ((retro_word - retro_neighbour) ** 2)) / args.batch_size
                loss_3 = args.gamma * torch.sum(cos(word, other) - cos(retro_word, retro_other))**2 / args.batch_size

                loss_1_list.append(loss_1)
                loss_2_list.append(loss_2)
                loss_3_list.append(loss_3)

                if losses == [1, 2, 3]:
                    loss = loss_1 + loss_2 + loss_3
                elif losses == [1, 2]:
                    loss = loss_1 + loss_2
                elif losses == [2, 3]:
                    loss = loss_2 + loss_3
                elif losses == [1]:
                    loss = loss_1
                elif losses == [2]:
                    loss = loss_2
                elif losses == [3]:
                    loss = loss_3
                else:
                    raise ValueError('Not implemented loss choosen.')

                loss.backward()
                opt.step()

                retro_emb_matrix[id_word_batch] = retro_word.data
                if args.train_var == "all":
                    retro_emb_matrix[id_other_batch] = retro_other.data
                    retro_emb_matrix[id_neighbour_batch] = retro_neighbour.data

                progress_bar(batch, no_batches, 'L: %.3f | L1: %.3f | L2: %.3f | L3: %.3f' % (loss, loss_1, loss_2, loss_3))

        wv_retro = {}
        for key in word_to_id.keys():
            wv_retro[key] = retro_emb_matrix[word_to_id[key]].numpy()

        return wv_retro, [[loss_1_list, "loss_1"], [loss_2_list, "loss_2"], [loss_3_list, "loss_3"]]


def retrofit():
    emb_words, emb_matrix_all, wv = load_embedding_from_h5(args.embedding)

    if args.verbose:
        print("evaluating on {} lexicon..".format(args.lexicon_name))

    lexicon = read_lexicon(os.path.join('src/scripts/retrofitting/lexicons', args.lexicon_name + '.txt'))

    #  either do retrofitting or use 2nd embedding
    if args.retrofitting:
        wv_2, losses = faruqui(wv, lexicon, args.n_epochs, args.losses, args.losses_2)
        calc_norm_of_wv(wv, wv_2, lexicon)
    else:
        _, _, wv_2 = load_embedding_from_h5(args.second_embedding)

    if args.sum:

        if args.q:
            Q = ortho_group.rvs(dim=300)
            wv_q = {}
            for key in wv_2.keys():
                wv_q[key] = Q @ wv_2[key]

            wv_2 = wv_q

        if args.pca:
            pca = PCA()
            pca.fit(emb_matrix_all)
            wv_2_emb_matrix, q_word_to_id, _ = dict_to_matrix_and_id(wv_2)
            z = pca.transform(wv_2_emb_matrix)
            z[:, 0:args.components] = 0
            wv_2_emb_matrix = pca.inverse_transform(z)

            if args.normalize_wv2:
                pca_qw = PCA()
                pca_qw.fit(wv_2_emb_matrix)
                print(pca.explained_variance_[0])
                print(pca_qw.explained_variance_[0])
                wv_2_emb_matrix = pca.explained_variance_[0] / pca_qw.explained_variance_[0] * wv_2_emb_matrix

            wv_2 = {}

            for key in q_word_to_id.keys():
                wv_2[key] = wv_2_emb_matrix[q_word_to_id[key]]

        for key in wv_2.keys():

            wv_2[key] = wv[key] + wv_2[key]


        if args.save_embedding:
            export_dict_to_h5(wv_2, os.path.join(DATA_DIR, "embeddings", args.save_text + ".h5"))
    else:

        if args.save_embedding:
            export_dict_to_h5(wv_2, os.path.join(DATA_DIR, "embeddings", args.save_text + ".h5"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--gamma", default=1, type=float)

    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--n-epochs", default=10, type=int)
    parser.add_argument("--components", default=20, type=int)

    parser.add_argument('--losses', nargs='+', default=[1, 2])
    parser.add_argument('--losses-2', nargs='+', default=[1, 2])

    parser.add_argument("--embedding", default='wiki', type=str)
    parser.add_argument('--lexicon-name', default='wordnet-synonyms+', type=str)
    parser.add_argument("--optimization", default='eiu', type=str, help='choose from sgd, adam, eiu.')
    parser.add_argument("--save-text", default='', type=str)
    parser.add_argument("--train-var", default='all', type=str)
    parser.add_argument("--second-embedding", default='', type=str)

    parser.add_argument("--normalize-wv2", action='store_true')
    parser.add_argument("--pca", action='store_true')
    parser.add_argument("--save-embedding", action='store_true')
    parser.add_argument("--sum", action='store_true')
    parser.add_argument("--q", action='store_true')
    parser.add_argument("--retrofitting", action='store_true')
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()
    retrofit()
