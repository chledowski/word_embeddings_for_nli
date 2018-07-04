"""
Few utilities
"""

import logging
import h5py
from logging import handlers
import datetime
from collections import OrderedDict
from copy import deepcopy
import socket
import numpy as np
from numpy import linalg as LA
from keras.utils import np_utils
from sklearn.decomposition import PCA
import re
import copy
import matplotlib
import os
import sys
import time
from src import DATA_DIR
from src.util.embedding import export_embedding_h5
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.evaluate import evaluate_similarity, evaluate_analogy
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_spectral_norm(matrix, k=10, centering=True):

    mean_matrix = np.mean(matrix, axis=0)

    if centering:
        matrix = matrix - mean_matrix

    pca = PCA()
    pca.fit(matrix)
    eigenvalues = pca.explained_variance_[:k]

    return eigenvalues


def dict_to_matrix_and_id(d):
    matrix = []
    word_to_id = {}
    id_to_word = {}
    i = 0

    for key in d.keys():
        matrix.append(d[key])
        word_to_id[key] = i
        id_to_word[i] = key
        i += 1

    return np.array(matrix), word_to_id, id_to_word


def evaluate_wv(wv, simlex_only=False):
    tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "SIMLEX999": fetch_SimLex999()
    }

    analogy_tasks = {
        # "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    results = {}

    if simlex_only:
        x = evaluate_similarity(wv, tasks["SIMLEX999"].X, tasks["SIMLEX999"].X)
        results["SIMLEX999"] = x
        print("Spearman correlation of scores on {} {}".format("SIMLEX999", x))
    else:
        for name, data in iteritems(tasks):
            x = evaluate_similarity(wv, data.X, data.y)
            results[name] = x
            print("Spearman correlation of scores on {} {}".format(name, x))

        for name, data in iteritems(analogy_tasks):
            x = evaluate_analogy(wv, data.X, data.y)
            results[name] = x
            print("Analogy prediction accuracy on {} {}".format(name, x))

    return results


def load_embedding_from_h5(name, normalize = False):
    path = os.path.join(DATA_DIR, 'embeddings', name + ".h5")
    emb_file = h5py.File(path, 'r')
    emb_words = emb_file['words_flatten'][0].split('\n')
    emb_matrix_all = emb_file[list(emb_file.keys())[0]][:]

    if normalize:
        emb_matrix_all = normalize_embeddings(emb_matrix_all)

    wv = {}
    for i in range(len(emb_words)):
        wv[emb_words[i]] = emb_matrix_all[i]

    return emb_words, emb_matrix_all, wv


def modified_stream(stream):
    def _stream():
        while True:
            it = stream.get_epoch_iterator()
            for x1, _, x2, _, y in it:
                yield [x1, x2], np_utils.to_categorical(y, 3)
    return _stream
#
# def normalize_embeddings(matrix):
#     wv = [matrix[i] for i in range(len(matrix[:,0]))]
#
#     for i in range(np.shape(matrix)[0]):
#         print (wv[i])
#         print( math.sqrt((wv[i] ** 2).sum() + 1e-6))
#         wv[i] /= math.sqrt((wv[i] ** 2).sum() + 1e-6)
#         # matrix[i] = matrix[i] / np.sqrt(np.sum(matrix[i]**2) + 1e-6)
#     return matrix


def normalize_embeddings(matrix):
    matrix = matrix.T / (np.linalg.norm(matrix, None, axis=1) + 1e-6)
    return matrix.T


def calc_loss(wv, lexicon, string):
    newWordVecs = deepcopy(wv)
    wvVocab = set(newWordVecs.keys())
    loopVocab = wvVocab.intersection(set(lexicon.keys()))

    loss = 0
    k = 0
    for word in loopVocab:
        wordNeighbours = set(lexicon[word]).intersection(wvVocab)
        numNeighbours = len(wordNeighbours)
        for ppWord in wordNeighbours:
            loss += sum((newWordVecs[ppWord] - newWordVecs[word])**2) / numNeighbours
        k += 1
        loss += sum((wv[word] - newWordVecs[word])**2)
    print("loss {}: ".format(string), loss/k)


def calc_norm_of_wv(wv_before, wv_after, lexicon):
    i = 0
    norm_sum = 0
    lexi_keys = set(lexicon.keys())
    for key in wv_before.keys():
        if key in lexi_keys:
            norm_sum += LA.norm(wv_after[key]) / LA.norm(wv_before[key])
            i += 1
    print("num of words in lexicon: ", i)
    print("avg norm of emb: ", norm_sum/i)


def plot(list_of_lists, savepath):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(list(range(len(list_of_lists[0][0]))), list_of_lists[0][0], label=list_of_lists[0][1], color='g', alpha=0.7)
    ax2.plot(list(range(len(list_of_lists[0][0]))), list_of_lists[1][0], label=list_of_lists[1][1], color='b', alpha=0.7)

    if len(list_of_lists) > 2:
        ax1.plot(list(range(len(list_of_lists[0][0]))), list_of_lists[2][0], label=list_of_lists[0][1], color='r', alpha=0.7)

    ax1.set_xlabel('batch')
    ax1.set_ylabel('loss_1', color='g')
    ax2.set_ylabel('loss_2', color='b')

    plt.legend()
    plt.savefig(savepath)
    plt.clf()


def plot3(list_of_lists, savepath):
    plt.plot(list_of_lists[2][0], label=list_of_lists[0][1], color='r', alpha=0.7)

    plt.savefig(savepath)
    plt.clf()


def remove_mean_and_d_components(matrix, D=0, centering=False, partial_whitening=False):

    mean_matrix = np.mean(matrix, axis=0)

    if centering:
        matrix = matrix - mean_matrix

    if D == 0:
        return matrix

    pca = PCA()
    pca.fit(matrix)
    components = pca.components_[:D]
    eigenvalues = pca.explained_variance_[:D]
    # print(eigenvalues)
    min_eigenvalue = min(eigenvalues)
    # print(min_eigenvalue)

    output = []
    for i in range(len(matrix)):
        sub = 0
        for j in range(D):
            if partial_whitening:
                sub += ((np.sqrt(eigenvalues[j]) - np.sqrt(min_eigenvalue))/np.sqrt(eigenvalues[j])) * np.dot(components[j], matrix[i])*components[j]
            else:
                sub += np.dot(components[j], matrix[i]) * components[j]
        v = matrix[i] - sub
        output.append(v)

    output = np.array(output)
    #
    if centering:
        output = output + mean_matrix
    #
    # mean_matrix = np.mean(output, axis=0)
    # output = output - mean_matrix

    # print("check1")
    # pca_check = PCA()
    # pca_check.fit(output)
    # print(pca_check.explained_variance_[:D])
    #
    # print("check2")
    # pca_check2 = PCA()
    # pca_check2.fit(output-matrix)
    # print(pca_check2.explained_variance_[:D+5])

    return output


def split_components(matrix, D):

    pca = PCA()
    pca.fit(matrix)
    components = pca.components_[:D]

    output = []
    pc_output = []
    for i in range(len(matrix)):
        sub = 0
        for j in range(D):
                sub += np.dot(components[j], matrix[i]) * components[j]
        v = matrix[i] - sub
        output.append(v)
        pc_output.append(sub)

    output = np.array(output)
    pc_output = np.array(pc_output)

    return output, pc_output, pca


def split_components_with_whitening(matrix, D):

    pca = PCA()
    pca.fit(matrix)
    components = pca.components_
    eigenvalues = pca.explained_variance_
    max_eigenvalue = max(eigenvalues)

    output = []
    pc_output = []
    for i in range(len(matrix)):
        sub = 0
        for j in range(D):
                sub += ((np.sqrt(eigenvalues[j]) - np.sqrt(max_eigenvalue)) / np.sqrt(eigenvalues[j])) * np.dot(
                    components[j], matrix[i]) * components[j]
        v = matrix[i] - sub
        output.append(v)
        pc_output.append(sub)

    output = np.array(output)
    pc_output = np.array(pc_output)

    print("check1")
    pca_check = PCA()
    pca_check.fit(output)
    print(pca_check.explained_variance_[:D])

    print("check2")
    pca_check2 = PCA()
    pca_check2.fit(output-matrix)
    print(pca_check2.explained_variance_[:D+5])

    return output, pc_output


def vocabulary_embedding_from_gensim_model(model):
    vocabulary = []
    embedding_matrix = []
    for key in model.wv.vocab:
        vocabulary.append(key)
        embedding_matrix.append(model.wv.get_vector(key).copy())
    return vocabulary, np.array(embedding_matrix)


def dict_from_gensim_model(model):
    dct = {}
    for key in model.wv.vocab:
        dct[key] = model.wv.get_vector(key).copy()
    return dct


def softmax(v, T):
    exp_v = np.exp(v/T)
    return exp_v / np.sum(exp_v)


def vec2str(vector):
    """Transforms a fixed size vector into a unicode string."""
    return "".join(map(chr, vector)).strip('\00')


isNumber = re.compile(r'\d+.*')


def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()


def read_lexicon(filename):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


def read_lexicon_symmetrically(filename):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]

    lexicon_symmetrical = copy.deepcopy(lexicon)
    for key in list(lexicon.keys()):
        for el in lexicon_symmetrical[key]:
            if not key in lexicon_symmetrical[el]:
                lexicon_symmetrical[el].append(key)

    return lexicon_symmetrical


def str2vec(str_, length):
    """Trasforms a string into a fixed size np.array
    Adds padding, if necessary. Truncates, if necessary.
    Importanty, if the input is a unicode string, the resulting
    array with contain unicode codes.
    """
    vector = np.array(list(map(ord, str_)))[:length]
    pad_length = max(0, length - len(str_))
    return np.pad(vector, (0, pad_length), 'constant')


def rename(var, name):
    var.name = name
    return var


def smart_sum(x):
    for i in range(x.ndim):
        x = x.sum(axis=-1)
    return x


def masked_root_mean_square(x, mask):
    """Masked root mean square for a 3D tensor"""
    return (smart_sum((x * mask[:, :, None]) ** 2) / x.shape[2] / mask.sum()) ** 0.5


def get_free_port():
    # Copy-paste from
    # http://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def kwargs_namer(**fnc_kwargs):
    return "_".join("{}={}".format(k, v) for k, v in OrderedDict(**fnc_kwargs).items() if k not in ['run_name'])


def utc_timestamp():
    return str(int(10 * (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()))


## Config logger

def parse_logging_level(logging_level):
    """
    :param logging_level: Logging level as string
    :return: Parsed logging level
    """
    lowercase = logging_level.lower()
    if lowercase == 'debug': return logging.DEBUG
    if lowercase == 'info': return logging.INFO
    if lowercase == 'warning': return logging.WARNING
    if lowercase == 'error': return logging.ERROR
    if lowercase == 'critical': return logging.CRITICAL
    raise ValueError('Logging level {} could not be parsed.'.format(logging_level))


def configure_logger(name = __name__,
                     console_logging_level = logging.INFO,
                     file_logging_level = logging.INFO,
                     log_file = None,
                     redirect_stdout=False,
                     redirect_stderr=False):
    """
    Configures logger
    :param name: logger name (default=module name, __name__)
    :param console_logging_level: level of logging to console (stdout), None = no logging
    :param file_logging_level: level of logging to log file, None = no logging
    :param log_file: path to log file (required if file_logging_level not None)
    :return instance of Logger class
    """
    if console_logging_level is None and file_logging_level is None:
        return # no logging

    if isinstance(console_logging_level, str):
        console_logging_level = parse_logging_level(console_logging_level)

    if isinstance(file_logging_level, str):
        file_logging_level = parse_logging_level(file_logging_level)

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    if file_logging_level is not None:
        if log_file is None:
            raise ValueError("If file logging enabled, log_file path is required")
        fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576*5), backupCount=7)
        fh.setFormatter(format)
        logger.addHandler(fh)

    logger.info("Logging configured!")

    if redirect_stderr:
        sys.stderr = LoggerWriter(logger.warning)
    if redirect_stdout:
        sys.stdout = LoggerWriter(logger.info)

    return logger

def copy_streams_to_file(log_file, stdout=True, stderr=True):
    logger = logging.getLogger("_copy_stdout_stderr_to_file")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(message)s")
    fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    logger.addHandler(fh)

    if stderr:
        sys.stderr = LoggerWriter(logger.warning)

    if stdout:
        sys.stdout = LoggerWriter(logger.info)


class LoggerWriter:
    """
    This class can be used when we want to redirect stdout or stderr to a logger instance.
    Example of usage:
    log = logging.getLogger('foobar')
    sys.stdout = LoggerWriter(log.debug)
    sys.stderr = LoggerWriter(log.warning)
    """
    def __init__(self, level, also_print=False):
        self.level = level
        self.also_print = also_print

    def write(self, message):
        if message != '\n':
            self.level(message)
        if self.also_print:
            print(message)

    def flush(self):
        pass


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def export_dict_to_h5(wv, path):
    retro_vocab = []
    retro_matrix = []

    for key in wv.keys():
        retro_vocab.append(key)
        retro_matrix.append(wv[key])

    export_embedding_h5(retro_vocab, np.array(retro_matrix),
                        output=path)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f