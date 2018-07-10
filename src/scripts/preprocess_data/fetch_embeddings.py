#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Small script to fetch required raw data, including publicly available pretrained embeddings:
 * GloVe
 * PDC
Saves in format that is compatible with models.phm.embeddings.*
"""
import os

_stdout_log = ""
_stderr_log = ""

import argparse
import logging
import subprocess
import sys
import threading
from six import string_types

from src import DATA_DIR
from src.util.embedding import export_embedding_h5, Embedding, _remove_chars

import pickle as pickle
from os import path

from web.embeddings import fetch_conceptnet_numberbatch, fetch_LexVec, fetch_SG_GoogleNews, \
    fetch_HDC, fetch_PDC

EMBEDDINGS_DIR = path.join(DATA_DIR, "raw", "embeddings")

logger = logging.getLogger(__name__)


def exec_command(command, flush_stdout=False, flush_stderr=False, cwd=None, timeout=None, verbose=1):
    global _stdout_log, _stderr_log
    _stdout_log, _stderr_log = "", ""

    if isinstance(command, string_types):
        command = [command]

    if verbose:
        logging.info("Running " + str(command))

    # timeout != None is supported only for Unix
    if timeout is not None or (flush_stdout == False and flush_stderr == False):
        import subprocess32
        p = subprocess32.Popen(
            command, stdout=subprocess32.PIPE, stderr=subprocess32.PIPE, cwd=cwd, shell=True, bufsize=0
        )

        assert flush_stdout == False, "Not supported flush_stdout"

        try:
            stdoutdata, stderrdata = p.communicate(timeout=timeout)
        except subprocess32.TimeoutExpired:
            return [], [], "timeout"

        if p.returncode != 0 and verbose:
            logging.error("Failed " + str(command))

        return stdoutdata.split("\n") if stdoutdata else [], \
               stderrdata.split("\n") if stderrdata else [], \
               p.returncode
    else:
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, shell=True, bufsize=0
        )

        out_thread = threading.Thread(name='stdout_thread', target=stdout_thread, args=(p, flush_stdout))
        err_thread = threading.Thread(name='stderr_thread', target=stderr_thread, args=(p, flush_stderr))

        err_thread.start()
        out_thread.start()

        out_thread.join()
        err_thread.join()

        p.wait()

        if p.returncode != 0 and verbose:
            logging.error("Failed " + str(command))

        return _stdout_log.split("\n"), _stderr_log.split("\n"), p.returncode


def stdout_thread(pipe, flush_stdout=False):
    global _stdout_log
    for line in pipe.stdout.readlines():
        _stdout_log += line + "\n"
        if flush_stdout:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


def stderr_thread(pipe, flush_stderr=False):
    global _stderr_log
    for line in pipe.stderr.readlines():
        _stderr_log += line + "\n"
        if flush_stderr:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()


def _fetch_file(url, destination):
    if not path.exists(destination):
        logging.info("Downloading " + url + " to " + destination)
        cmd = "wget {} -O {}".format(url, destination)
        _, _, ret = exec_command(cmd, flush_stdout=True)
        if ret != 0:
            raise RuntimeError("Failed wget {}".format(cmd))
    else:
        logger.info("Already downloaded " + destination)


def load_embedding(fname, format="word2vec_bin", normalize=True,
                   lower=False, clean_words=False, load_kwargs={}):
    """
    Loads embeddings from file

    Parameters
    ----------
    fname: string
      Path to file containing embedding

    format: string
      Format of the embedding. Possible values are:
      'word2vec_bin', 'word2vec', 'glove', 'dict'

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.
    """
    assert format in ['word2vec_bin', 'word2vec', 'glove', 'dict'], "Unrecognized format"
    if format == "word2vec_bin":
        w = Embedding.from_word2vec(fname, binary=True)
    elif format == "word2vec":
        w = Embedding.from_word2vec(fname, binary=False)
    elif format == "glove":
        w = Embedding.from_glove(fname, **load_kwargs)
    elif format == "dict":
        d = pickle.load(open(fname, "rb"))
        w = Embedding.from_dict(d)
    if normalize:
        w.normalize_words(inplace=True)
    if lower or clean_words:
        w.standardize_words(lower=lower, clean_words=clean_words, inplace=True)
    return w


def fetch_glove(dim=300, corpus="wiki", normalize=False, lower=False, clean_words=True):
    """
    Fetches GloVe embeddings.

    Parameters
    ----------
    dim: int, default: 300
      Dimensionality of embedding (usually performance increases with dimensionality).
      Available dimensionalities:
        * wiki: 50, 100, 200, 300
        * common-crawl-42B: 300
        * common-crawl-840B: 300
        * twitter: 25, 50, 100, 200

    corpus: string, default: "wiki"
      Corpus that GloVe vector were trained on.
      Available corpuses: "wiki", "common-crawl-42B", "common-crawl-840B", "twitter-27B"

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.

    Returns
    -------
    w: Embedding
      Embedding instance

    References
    ----------
    Project website: http://nlp.stanford.edu/projects/glove/

    Notes
    -----
    Loading GloVe format can take a while
    """
    download_file = {
        "wiki": "http://nlp.stanford.edu/data/glove.6B.zip",
        "common-crawl-42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
        "common-crawl-840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter-27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    }

    embedding_file = {
        "wiki": {
            50: "glove.6B/glove.6B.50d.txt",
            100: "glove.6B/glove.6B.100d.txt",
            200: "glove.6B/glove.6B.200d.txt",
            300: "glove.6B/glove.6B.300d.txt"
        },
        "common-crawl-42B": {
            300: "glove.42B.300d/glove.42B.300d.txt"
        },
        "common-crawl-840B": {
            300: "glove.840B.300d/glove.840B.300d.txt"
        },
        "twitter-27B": {
            25: "glove.twitter.27B/glove.twitter.27B.25d.txt",
            50: "glove.twitter.27B/glove.twitter.27B.50d.txt",
            100: "glove.twitter.27B/glove.twitter.27B.100d.txt",
            200: "glove.twitter.27B/glove.twitter.27B.200d.txt",
        }
    }

    vocab_size = {
        "wiki": 400000,
        "common-crawl-42B": 1917494,
        "common-crawl-840B": 2196017,
        "twitter-27B": 1193514
    }

    assert corpus in download_file, "Unrecognized corpus"
    assert dim in embedding_file[corpus], "Not available dimensionality"

    dest = path.join(EMBEDDINGS_DIR, embedding_file[corpus][dim])

    if not os.path.exists(dest):
        dest_zip = path.join(EMBEDDINGS_DIR, path.dirname(embedding_file[corpus][dim]) + ".zip")

        if not os.path.exists(path.dirname(dest_zip)):
            os.system("mkdir -p " + path.dirname(dest_zip))

        _fetch_file(url=download_file[corpus], destination=dest_zip)
        ret = os.system("unzip " + dest_zip + " -d " + path.dirname(dest))
        if ret != 0:
            raise Exception("Failed unzipping")
        os.system("rm " + dest_zip)

    return load_embedding(path.join(EMBEDDINGS_DIR, embedding_file[corpus][dim]),
                          format="glove",
                          normalize=normalize,
                          lower=lower, clean_words=clean_words,
                          load_kwargs={"vocab_size": vocab_size[corpus], "dim": dim})


def kwargs_to_name(**obj):
    current_name = []
    for k in sorted(obj):
        # TODO: Handle floats and non-standard object types better
        current_name.append(k + "=" + str(obj[k]))
    return _remove_chars("_".join(current_name), old='/', new='%', meta='_')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetching pretrained embeddings.')
    parser.add_argument('--embeddings', nargs='+', default=['wiki', 'glove', 'all'], type=str, help='lr schedule')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(DATA_DIR, 'embeddings')):
        os.makedirs(os.path.join(DATA_DIR, 'embeddings'))

    if 'all' in args.embeddings:

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "wiki" + ".h5")):
            print("Fetching GloVe 6B")
            E = fetch_glove(corpus="wiki")
            print("Saving embeddings")
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "wiki" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "common-crawl-42B" + ".h5")):
            print("Fetching GloVe 42B")
            E = fetch_glove(corpus="common-crawl-42B")
            print("Saving embeddings")
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "common-crawl-42B" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "common-crawl-840B" + ".h5")):
            print("Fetching GloVe 840B")
            E = fetch_glove(corpus="common-crawl-840B")
            print("Saving embeddings")
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "common-crawl-840B" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "conceptnet" + ".h5")):
            print("Fetching ConceptNet")
            E = fetch_conceptnet_numberbatch()
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "conceptnet" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "lexvec" + ".h5")):
            print("Fetching LexVec")
            E = fetch_LexVec()
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "lexvec" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "sg_googlenews" + ".h5")):
            print("Fetching SG_GoogleNews")
            E = fetch_SG_GoogleNews()
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "sg_googlenews" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "hdc" + ".h5")):
            print("Fetching HDC")
            E = fetch_HDC()
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "hdc" + ".h5"))

        if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "pdc" + ".h5")):
            print("Fetching PDC")
            E = fetch_PDC()
            export_embedding_h5(E.vocabulary.words, E.vectors,
                                output=os.path.join(DATA_DIR, "embeddings", "pdc" + ".h5"))

    else:

        if 'glove' in args.embeddings:

            if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "wiki" + ".h5")):
                print("Fetching GloVe 6B")
                E = fetch_glove(corpus="wiki")
                print("Saving embeddings")
                export_embedding_h5(E.vocabulary.words, E.vectors,
                                    output=os.path.join(DATA_DIR, "embeddings", "wiki" + ".h5"))

            if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "common-crawl-42B" + ".h5")):
                print("Fetching GloVe 42B")
                E = fetch_glove(corpus="common-crawl-42B")
                print("Saving embeddings")
                export_embedding_h5(E.vocabulary.words, E.vectors,
                                    output=os.path.join(DATA_DIR, "embeddings", "common-crawl-42B" + ".h5"))

            if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "common-crawl-840B" + ".h5")):
                print("Fetching GloVe 840B")
                E = fetch_glove(corpus="common-crawl-840B")
                print("Saving embeddings")
                export_embedding_h5(E.vocabulary.words, E.vectors,
                                    output=os.path.join(DATA_DIR, "embeddings", "common-crawl-840B" + ".h5"))

        elif 'wiki' in args.embeddings:

            if not os.path.exists(os.path.join(DATA_DIR, "embeddings", "wiki" + ".h5")):
                print("Fetching GloVe 6B")
                E = fetch_glove(corpus="wiki")
                print("Saving embeddings")
                export_embedding_h5(E.vocabulary.words, E.vectors,
                                    output=os.path.join(DATA_DIR, "embeddings", "wiki" + ".h5"))