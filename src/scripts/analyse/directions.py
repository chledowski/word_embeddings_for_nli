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


def analyse():
    emb_words_1, emb_matrix_all_1, wv_1 = load_embedding_from_h5(args.embedding_1)
    emb_words_2, emb_matrix_all_2, wv_2 = load_embedding_from_h5(args.embedding_2)

    if args.q:
        # Q = ortho_group.rvs(dim=300)
        # wv_q = {}
        # for key in wv_2.keys():
        #     wv_q[key] = Q @ wv_2[key]
        # emb_matrix_all_2, _ , _ = dict_to_matrix_and_id(wv_q)
        #
        Q = ortho_group.rvs(dim=300)
        for i in range(len(emb_matrix_all_2)):
            emb_matrix_all_2[i] = Q @ emb_matrix_all_2[i]

    pca_1 = PCA()
    pca_1.fit(emb_matrix_all_1)
    z_1 = pca_1.components_[:args.components]

    pca_2 = PCA()
    pca_2.fit(emb_matrix_all_2)
    z_2 = pca_2.components_[:args.components]

    m = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            m[i, j] = np.dot(z_1[i], z_2[j])

    print(m)


    #
    #
    #
    # if args.sum:
    #
    #     if args.q:
    #         Q = ortho_group.rvs(dim=300)
    #         wv_q = {}
    #         for key in wv_2.keys():
    #             wv_q[key] = Q @ wv_2[key]
    #
    #         wv_2 = wv_q
    #
    #     if args.pca:
    #         pca = PCA()
    #         pca.fit(emb_matrix_all)
    #         wv_2_emb_matrix, q_word_to_id, _ = dict_to_matrix_and_id(wv_2)
    #         z = pca.transform(wv_2_emb_matrix)
    #         z[:, 0:args.components] = 0
    #         wv_2_emb_matrix = pca.inverse_transform(z)
    #
    #         if args.normalize_wv2:
    #             pca_qw = PCA()
    #             pca_qw.fit(wv_2_emb_matrix)
    #             print(pca.explained_variance_[0])
    #             print(pca_qw.explained_variance_[0])
    #             wv_2_emb_matrix = pca.explained_variance_[0] / pca_qw.explained_variance_[0] * wv_2_emb_matrix
    #
    #         wv_2 = {}
    #
    #         for key in q_word_to_id.keys():
    #             wv_2[key] = wv_2_emb_matrix[q_word_to_id[key]]
    #
    #     for key in wv_2.keys():
    #
    #         wv_2[key] = wv[key] + wv_2[key]
    #
    #
    #     if args.save_embedding:
    #         export_dict_to_h5(wv_2, os.path.join(DATA_DIR, "embeddings", args.save_text + ".h5"))
    # else:
    #
    #     if args.save_embedding:
    #         export_dict_to_h5(wv_2, os.path.join(DATA_DIR, "embeddings", args.save_text + ".h5"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--components", default=5, type=int)
    parser.add_argument("--embedding-1", default='gcc840', type=str)
    parser.add_argument("--embedding-2", default='fq12', type=str)

    parser.add_argument("--normalize-wv2", action='store_true')
    parser.add_argument("--pca", action='store_true')
    parser.add_argument("--save-embedding", action='store_true')
    parser.add_argument("--sum", action='store_true')
    parser.add_argument("--q", action='store_true')
    parser.add_argument("--retrofitting", action='store_true')
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()
    analyse()
