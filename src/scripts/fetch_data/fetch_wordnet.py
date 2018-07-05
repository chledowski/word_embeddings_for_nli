import sys
import numpy
from six.moves import cPickle as pkl
import re
import os
import tarfile
from collections import OrderedDict

sys.setrecursionlimit(1000)
from src import DATA_DIR


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath


def download_wordnet(dirpath):
    if os.path.exists(dirpath):
        print('Found WordNet 3.0 - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://wordnetcode.princeton.edu/3.0/WNprolog-3.0.tar.gz'
    ungzip(download(url, dirpath))


def ungzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with tarfile.open(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def build_dictionary_wordnet(filepaths, dst_path=None, lowercase=False, remove_phrase=True):
    word_id_num = OrderedDict()
    id_word = OrderedDict()
    id_num_word = OrderedDict()
    count_phrase = 0
    for filepath in filepaths:
        print('Processing', filepath)
        with open(filepath, 'r') as f:
            # format: s(100001740,1,'entity',n,1,11).
            for line in f:
                matchObj = re.match(r'^s\((.*)\)', line)
                s_list = matchObj.group(1).split(',')
                word = s_list[2][1:-1]
                synset_id = s_list[0]
                w_num = s_list[1]
                id_num = synset_id + '.' + w_num
                if ' ' in word:
                    if remove_phrase:
                        continue
                    else:
                        word_list = word.split()
                        word = '_'.join(word_list)
                        count_phrase += 1
                if lowercase:
                    word = word.lower()
                if word not in word_id_num:
                    word_id_num[word] = set([id_num])
                else:
                    word_id_num[word].add(id_num)

                if synset_id not in id_word:
                    id_word[synset_id] = set([word])
                else:
                    id_word[synset_id].add(word)

                if id_num not in id_num_word:
                    id_num_word[id_num] = set([word])
                else:
                    id_num_word[id_num].add(word)

    if dst_path:
        with open(dst_path, 'wb') as f:
            pkl.dump(word_id_num, f)
            pkl.dump(id_word, f)
            pkl.dump(id_num_word, f)

    print('number of phrases', count_phrase)
    print('size of word dictionary', len(word_id_num))
    print('size of synset_id dictionary', len(id_word))
    print('size of synset_id_num dictionary', len(id_num_word))

    return word_id_num, id_word, id_num_word


def read_synonymy(id_word):
    w_w_features = OrderedDict()
    for key in id_word.keys():
        for w1 in id_word[key]:
            for w2 in id_word[key]:
                # if w1 == w2:
                # continue
                w_w_features[w1 + ';' + w2] = 1

    return w_w_features


def add_recursive(dictionary, key, n):
    seqs = []
    if key not in dictionary or n > 7:
        pass
    else:
        for k in dictionary[key]:
            seqs.extend(add_recursive(dictionary, k, n + 1))
            temp = k + '_' + str(n)
            seqs = [temp] + seqs
    return seqs


def read_hyper_hypo(file, id_word):
    w_w_features = OrderedDict()
    w_w_features_reflexive = OrderedDict()
    id1_id2 = OrderedDict()
    id2_id1 = OrderedDict()

    with open(file, 'r') as f:
        for line in f:
            matchObj = re.match(r'^\w+\((.*)\)', line)
            p_list = matchObj.group(1).split(',')
            if p_list[0] not in id1_id2:
                id1_id2[p_list[0]] = set([p_list[1]])
            else:
                id1_id2[p_list[0]].add(p_list[1])

            if p_list[1] not in id2_id1:
                id2_id1[p_list[1]] = set([p_list[0]])
            else:
                id2_id1[p_list[1]].add(p_list[0])

    w_w_features_same_parent = OrderedDict()
    for k, v in id2_id1.items():
        for vv1 in v:
            for vv2 in v:
                if vv1 != vv2:
                    if vv1 in id_word and vv2 in id_word:
                        w1s = id_word[vv1]
                        w2s = id_word[vv2]
                        for w1 in w1s:
                            for w2 in w2s:
                                if w1 == w2:
                                    continue
                                w_w_features_same_parent[w1 + ';' + w2] = 1

    new_id_id = OrderedDict()
    for k, v in id1_id2.items():
        seqs = []
        seqs.extend(add_recursive(id1_id2, k, 1))
        new_id_id[k] = set(seqs)

    for k, v in new_id_id.items():
        for vv in v:
            vv_id, vv_n = vv.split('_')
            if vv_id in id_word and k in id_word:
                w1s = id_word[k]
                w2s = id_word[vv_id]
                for w1 in w1s:
                    for w2 in w2s:
                        if w1 == w2:
                            continue
                        w_w_features[w1 + ';' + w2] = 1 - float(vv_n) / 8
                        w_w_features_reflexive[w2 + ';' + w1] = 1 - float(vv_n) / 8

    return w_w_features, w_w_features_reflexive, w_w_features_same_parent


def read_antony(file, id_num_word, reflexive=True):
    w_w_features = OrderedDict()
    if reflexive:
        w_w_features_reflexive = OrderedDict()
    with open(file, 'r') as f:
        for line in f:
            matchObj = re.match(r'^\w+\((.*)\)', line)
            p_list = matchObj.group(1).split(',')
            id_num_1 = p_list[0] + '.' + p_list[1]
            id_num_2 = p_list[2] + '.' + p_list[3]

            if id_num_1 in id_num_word and id_num_2 in id_num_word:
                w1s = id_num_word[id_num_1]
                w2s = id_num_word[id_num_2]
                for w1 in w1s:
                    for w2 in w2s:
                        if w1 == w2:
                            continue
                        w_w_features[w1 + ';' + w2] = 1
                        if reflexive:
                            w_w_features_reflexive[w2 + ';' + w1] = 1

    if reflexive:
        return w_w_features, w_w_features_reflexive
    else:
        return w_w_features


def features2pkl(feat_path, dict_path, out_path):
    bk_for_x = {}
    numpy.random.seed(1234)

    with open(feat_path, 'r') as f1:
        with open(dict_path, 'r') as f2:
            worddicts = pkl.load(f2)
            for line in f1:
                l = line.strip().split(' ')
                ids = l[0].split(';')
                if ids[0] in worddicts and ids[1] in worddicts:
                    ids0 = worddicts[ids[0]]
                    ids1 = worddicts[ids[1]]

                    if int(ids0) in bk_for_x:
                        bk_for_x[int(ids0)][int(ids1)] = map(float, l[1:])
                    else:
                        bk_for_x[int(ids0)] = {int(ids1): map(float, l[1:])}

    with open(out_path, 'wb') as f:
        pkl.dump(bk_for_x, f)

    count = 0
    for k, v in bk_for_x.items():
        count += len(v.keys())

    print('feature2pkl size', len(bk_for_x.keys()), count)


dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}


def build_dictionary(filepaths, dst_path, lowercase=False, wordnet=None, remove_phrase=True):
    word_freqs = OrderedDict()

    for k in wordnet.keys():
        if remove_phrase:
            if '_' in k:
                continue
        word_freqs[k] = 0

    for filepath in filepaths:
        print('Processing', filepath)
        with open(filepath, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['_PAD_'] = 0  # default, padding
    worddict['_UNK_'] = 1  # out-of-vocabulary
    worddict['_BOS_'] = 2  # begin of sentence token
    worddict['_EOS_'] = 3  # end of sentence token

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 4

    with open(dst_path, 'wb') as f:
        pkl.dump(worddict, f)

    print('dict size', len(worddict))


def build_sequence(filepath, dst_dir):
    filename = os.path.basename(filepath)
    print(filename)
    len_p = []
    len_h = []
    with open(filepath) as f, \
            open(os.path.join(dst_dir, 'premise_%s' % filename), 'w') as f1, \
            open(os.path.join(dst_dir, 'hypothesis_%s' % filename), 'w') as f2, \
            open(os.path.join(dst_dir, 'label_%s' % filename), 'w') as f3:
        next(f)  # skip the header row
        for line in f:
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(', ')')]
            f1.write(' '.join(words_in) + '\n')
            len_p.append(len(words_in))

            words_in = sents[2].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(', ')')]
            f2.write(' '.join(words_in) + '\n')
            len_h.append(len(words_in))

            f3.write(dic[sents[0]] + '\n')

    print('max min len premise', max(len_p), min(len_p))
    print('max min len hypothesis', max(len_h), min(len_h))


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def ordered_dicts_to_set(dicts):
    s = dict()
    for d in dicts:
        for k, _ in d.items():
            a, b = k.split(';')
            # if k.split(';')[0] == 'abstraction':
            #     print(k.split(';'))
            if a not in s:
                s[a] = set()
            s[a].add(b)

    return s

def pairs_to_txt(s, filename):
    with open(filename, "w+") as f:
        keys = list(s.keys())
        for i in range(len(keys)):
            word = keys[i]
            words = s[word]
            f.write(word + ' ' + ' '.join(words) + '\n')
            # i += 1
            # if l_1[0] not in used:
            #     line = l_1[0] + " "
            #     used.add(l_1[0])
            #     for l_2 in s:
            #         if l_2[0] == l_1[0]:
            #             line += l_2[1] + " "
            #     f.write(line + "\n")

if __name__ == '__main__':
    download_wordnet(os.path.join(DATA_DIR, 'wordnet'))

    wordnet_dir = os.path.join(DATA_DIR, 'wordnet/prolog')

    print('=' * 80)
    print('Preprocessing WordNet prolog and SNLI dataset')
    print('=' * 80)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dst_dir = os.path.join(base_dir, 'sequence_and_features')
    make_dirs([dst_dir])

    print('1. build dictionary of WordNet\n')
    word_id_num, id_word, id_num_word = build_dictionary_wordnet([os.path.join(wordnet_dir, 'wn_s.pl')],
                                                                 remove_phrase=True)

    print('2. obtain relation features\n')
    hypernymy, hyponymy, co_hyponyms = read_hyper_hypo(os.path.join(wordnet_dir, 'wn_hyp.pl'), id_word)
    print('hyponymy:', len(hyponymy))
    print('co_hyponyms', len(co_hyponyms))

    antonymy = read_antony(os.path.join(wordnet_dir, 'wn_ant.pl'), id_num_word, reflexive=False)
    print('antonymy:', len(antonymy))

    synonymy = read_synonymy(id_word)
    print('synonymy:', len(synonymy))
    s = ordered_dicts_to_set([hyponymy, hypernymy, antonymy, synonymy])
    pairs_to_txt(s, 'src/scripts/retrofitting/lexicons/kim.txt')
    s = ordered_dicts_to_set([hyponymy, hypernymy, antonymy, synonymy, co_hyponyms])
    pairs_to_txt(s, 'src/scripts/retrofitting/lexicons/cokim.txt')