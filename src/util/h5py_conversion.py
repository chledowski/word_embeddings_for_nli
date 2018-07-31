import h5py
import json
import logging
import numpy as np
import os
import pandas as pd
import tempfile
import tqdm
import traceback

from fuel.datasets.hdf5 import H5PYDataset
from nltk.tokenize import TreebankWordTokenizer
from src.util.corenlp import StanfordCoreNLP
from src import DATA_DIR

logger = logging.getLogger()


def _find_sublist(list_, sublist):
    indices = []
    for i in range(len(list_) - len(sublist) + 1):
        found = True
        for j in range(len(sublist)):
            if list_[i + j] != sublist[j]:
                found = False
                break
        if found:
            indices.append(i)
    return indices


def text_to_h5py_dataset(text_path, dst_path):
    # The simplest is to load everything to memory first.
    # If memory becomes an issue, this code can be optimized.
    words = []
    with open(text_path, 'r') as src:
        for line in src:
            words.extend(line.strip().split())

    with h5py.File(dst_path, 'w') as dst:
        dtype = h5py.special_dtype(vlen=bytes)
        table = dst.create_dataset('words', (len(words),), dtype=dtype)
        table[:] = words

        dst.attrs['split'] = H5PYDataset.create_split_array({
                'train' : {
                    'words' : (0, len(words))
                }
            })


def squad_to_h5py_dataset(squad_path, dst_path, corenlp_url):
    data = json.load(open(squad_path))
    data = data['data']

    text = []
    def add_text(list_):
        text.extend(list_)
        return len(text) - len(list_), len(text)

    corenlp = StanfordCoreNLP(corenlp_url)
    def tokenize(str_):
        annotations = json.loads(
            corenlp.annotate(str_,
                             properties={'annotators': 'tokenize,ssplit'}))
        tokens = []
        positions = []
        for sentence in annotations['sentences']:
            for token in sentence['tokens']:
                tokens.append(token['originalText'])
                positions.append(token['characterOffsetBegin'])
        return tokens, positions


    all_contexts = []
    all_questions = []
    all_q_ids = []
    all_answer_begins = []
    all_answer_ends = []

    num_issues = 0
    for article in data:
        for paragraph in article['paragraphs']:
            context, context_positions = tokenize(paragraph['context'])
            context_begin, context_end = add_text(context)

            for qa in paragraph['qas']:
                try:
                    question, _ = tokenize(qa['question'])
                    question_begin, question_end = add_text(question)
                    answer_begins = []
                    answer_ends = []

                    for answer in qa['answers']:
                        start = answer['answer_start']
                        assert paragraph['context'][start:start + len(answer['text'])] == answer['text']
                        answer_text, _ = tokenize(answer['text'])
                        begin = context_positions.index(answer['answer_start'])

                        end = begin + len(answer_text)
                        answer_begins.append(begin)
                        answer_ends.append(end)

                    all_contexts.append((context_begin, context_end))
                    all_questions.append((question_begin, question_end))
                    all_q_ids.append(qa['id'])
                    all_answer_begins.append(answer_begins)
                    all_answer_ends.append(answer_ends)
                except ValueError:
                    logger.error("tokenized context: {}".format(list(zip(context, context_positions))))
                    logger.error("qa: {}".format(qa))
                    traceback.print_exc()
                    num_issues += 1
    if num_issues:
        logger.error("there were {} issues".format(num_issues))

    num_examples = len(all_contexts)

    dst = h5py.File(dst_path, 'w')
    unicode_dtype = h5py.special_dtype(vlen=str)
    dst.create_dataset('text', (len(text),), unicode_dtype)
    dst.create_dataset('contexts', (num_examples, 2), 'int64')
    dst.create_dataset('questions', (num_examples, 2), 'int64')
    dst.create_dataset('q_ids', (num_examples,), unicode_dtype)
    vlen_int64 = h5py.special_dtype(vlen='int64')
    dst.create_dataset('answer_begins', (num_examples,), vlen_int64)
    dst.create_dataset('answer_ends', (num_examples,), vlen_int64)
    dst['text'][:] = text
    dst['contexts'][:] = all_contexts
    dst['questions'][:] = all_questions
    dst['q_ids'][:] = all_q_ids
    dst['answer_begins'][:] = all_answer_begins
    dst['answer_ends'][:] = all_answer_ends
    dst.attrs['split'] = H5PYDataset.create_split_array({
            'all' : {
                'contexts' : (0, num_examples),
                'questions' : (0, num_examples),
                'answer_begins' : (0, num_examples),
                'answer_ends' : (0, num_examples),
                'q_ids' : (0, num_examples)
            }
        })
    dst.close()


def add_words_ids_to_squad(h5_file, vocab):
    """Digitizes test with a vocabulary.
    Also saves the vocabulary into the hdf5 file.
    """
    with h5py.File(h5_file, 'a') as dst:
        unicode_dtype = h5py.special_dtype(vlen=str)
        dst.create_dataset('text_ids', (dst['text'].shape[0],), 'int64')
        dst.create_dataset('vocab_words', (vocab.size(),), unicode_dtype)
        dst.create_dataset('vocab_freqs', (vocab.size(),), 'int64')
        dst['text_ids'][:] = list(map(vocab.word_to_id, dst['text'][:]))
        dst['vocab_words'][:] = vocab.words
        dst['vocab_freqs'][:] = vocab.frequencies


### SNLI ###

SNLI_LABEL2INT = {"neutral": 1, "entailment": 2, "contradiction": 0}

def add_word_ids_to_snli(h5_file, vocab):
    with h5py.File(h5_file, 'a') as dst:
        N = len(dst['sentence1'])
        assert len(dst['sentence2']) == N

        dst.create_dataset('vocab_words', (vocab.size(),),  h5py.special_dtype(vlen=str))
        dst.create_dataset('vocab_freqs', (vocab.size(),), 'int64')
        dst['vocab_words'][:] = vocab.words
        dst['vocab_freqs'][:] = vocab.frequencies

        dtype = h5py.special_dtype(vlen=np.dtype('int32'))
        sentence1_ds = dst.create_dataset('sentence1_ids', (N, ), dtype=dtype)
        sentence2_ds = dst.create_dataset('sentence2_ids', (N, ), dtype=dtype)

        ### h5py nonsense ###
        sentence1_ds_shapes = dst.create_dataset('sentence1_ids_shapes', (N, 1), dtype=("int"))
        sentence2_ds_shapes = dst.create_dataset('sentence2_ids_shapes', (N, 1), dtype=("int"))
        ds_shape_labels = dst.create_dataset('ds_ids_shape_labels', (1, ), dtype=("S20"))
        ### h5py nonsense ###

        sentence1_ds[:] = np.array([[vocab.word_to_id(w) for w in s] for s in dst['sentence1'][:]])
        sentence2_ds[:] = np.array([[vocab.word_to_id(w) for w in s] for s in dst['sentence2'][:]])

        ### h5py nonsense ###
        sentence1_ds_shapes[:] = np.array([np.array(x).shape for x in dst['sentence1'][:]])
        sentence2_ds_shapes[:] = np.array([np.array(x).shape for x in dst['sentence2'][:]])
        ds_shape_labels[:] = np.array(['sentence_len'])

        sentence1_ds.dims.create_scale(sentence1_ds_shapes, 'shapes')
        sentence1_ds.dims[0].attach_scale(sentence1_ds_shapes)
        sentence1_ds.dims.create_scale(ds_shape_labels, 'shape_labels')
        sentence1_ds.dims[0].attach_scale(ds_shape_labels)

        sentence2_ds.dims.create_scale(sentence2_ds_shapes, 'shapes')
        sentence2_ds.dims[0].attach_scale(sentence2_ds_shapes)
        sentence2_ds.dims.create_scale(ds_shape_labels, 'shape_labels')
        sentence2_ds.dims[0].attach_scale(ds_shape_labels)
        ### h5py nonsense ###

        dst.attrs['split'] = H5PYDataset.create_split_array({
            'all': {
                'sentence1': (0, N),
                'sentence2': (0, N),
                'sentence1_ids': (0, N),
                'sentence2_ids': (0, N),
                'label': (0, N),
                'text': (0, len(dst['text']))
            }
        })


def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()


def remove_non_ascii_characters(parse):
    return parse.encode('ascii', errors='ignore').decode()


def lemmatize(file_path):
    tokenize_and_lemmatize_path = '%s/tokenize_and_lemmatize' % os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists('%s.class' % tokenize_and_lemmatize_path):
        print('Compile ...')
        cmd = 'javac -cp "%s/corenlp/stanford-corenlp-full-2016-10-31/*" %s.java' % (
            DATA_DIR, tokenize_and_lemmatize_path)
        print(cmd)
        os.system(cmd)
    print('Run ...')
    base_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_name_lemma = base_name + '_lemma.txt'
    out_path_lemma = os.path.join(base_dir, out_name_lemma)
    cmd = 'java -cp ".:{}/corenlp/stanford-corenlp-full-2016-10-31/*:{}" tokenize_and_lemmatize {} {}'.format(
        DATA_DIR, os.path.dirname(os.path.abspath(__file__)), file_path, out_path_lemma)
    print(cmd)
    os.system(cmd)
    return out_path_lemma


def snli_to_h5py_dataset(snli_path, dst_path, lowercase=False):
    logging.info("Reading CSV file")
    d = pd.read_csv(snli_path, sep="\t", error_bad_lines=False)

    # Remove NaN
    d = d[d['sentence2_binary_parse'].apply(lambda x: isinstance(x, str))]

    total_num_non_ascii_characters = 0
    total_num_sentences_with_no_ascii = 0
    for column in ['sentence1_binary_parse', 'sentence2_binary_parse']:
        total_num_non_ascii_characters += sum([
            len(s) - len(remove_non_ascii_characters((s))) for s in d[column]
        ])
        total_num_sentences_with_no_ascii += sum([
            len(s) - len(remove_non_ascii_characters((s))) > 0 for s in d[column]
        ])
        d[column] = d[column].apply(remove_non_ascii_characters)

    logging.info("Total num. of non-ASCII characters: %d" % total_num_non_ascii_characters)
    logging.info("Total num. of sentences with non-ASCII: %d" % total_num_sentences_with_no_ascii)

    # Remove labels without consensus
    d = d.drop(d.query('gold_label == "-"').index)

    # Add fields
    d['sentence1_tokenized'] = [[w.lower() if lowercase else w for w in extract_tokens_from_binary_parse(s)]
        for s in tqdm.tqdm(d['sentence1_binary_parse'], total=len(d))]
    d['sentence2_tokenized'] = [[w.lower() if lowercase else w for w in extract_tokens_from_binary_parse(s)]
        for s in tqdm.tqdm(d['sentence2_binary_parse'], total=len(d))]
    d['gold_label_int'] = [SNLI_LABEL2INT[x] for x in d['gold_label']]

    for column in ['sentence1_tokenized', 'sentence2_tokenized']:
        fp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        for s in d[column]:
            fp.write(' '.join(s) + '\n')
        fp.close()
        lemma_path = lemmatize(fp.name)
        with open(lemma_path) as lemma_fp:
            lemma_column = '%s_lemmatized' % column.split('_')[0]
            d[lemma_column] = [line.split() for line in lemma_fp]
        os.remove(fp.name)
        os.remove(lemma_path)

    # Get all words
    sentences = [extract_tokens_from_binary_parse(s) for s in d['sentence1_binary_parse']]
    sentences += [extract_tokens_from_binary_parse(s) for s in d['sentence2_binary_parse']]

    words = np.array([w.lower() if lowercase else w for s in tqdm.tqdm(sentences, total=len(sentences)) for w in s], dtype='S20')
    sentences = [] # For safety
    logging.info("Found {} words".format(len(words)))


    # Pack (I hate writing this h5py code)
    dtype = h5py.special_dtype(vlen='S20')
    dst = h5py.File(dst_path, "w")
    sentence1_ds = dst.create_dataset('sentence1', (len(d),), dtype=dtype)
    sentence2_ds = dst.create_dataset('sentence2', (len(d),), dtype=dtype)
    sentence1_lemma_ds = dst.create_dataset('sentence1_lemmatized', (len(d),), dtype=dtype)
    sentence2_lemma_ds = dst.create_dataset('sentence2_lemmatized', (len(d),), dtype=dtype)
    label_ds = dst.create_dataset('label', (len(d),), dtype=np.int32)
    words_ds = dst.create_dataset('text', (len(words),), dtype='S20')

    ### h5py nonsense ###
    sentence1_ds_shapes = dst.create_dataset('sentence1_ds_shapes', (len(d), 1), dtype=("int"))
    sentence2_ds_shapes = dst.create_dataset('sentence2_ds_shapes', (len(d), 1), dtype=("int"))
    sentence1_lemma_ds_shapes = dst.create_dataset('sentence1_lemma_ds_shapes', (len(d), 1), dtype=("int"))
    sentence2_lemma_ds_shapes = dst.create_dataset('sentence2_lemma_ds_shapes', (len(d), 1), dtype=("int"))
    ds_shape_labels = dst.create_dataset('ds_shape_labels', (1,), dtype="S20")
    ### h5py nonsense ###
    sentence1_ds[:] = np.array(d['sentence1_tokenized'])
    sentence2_ds[:] = np.array(d['sentence2_tokenized'])
    sentence1_lemma_ds[:] = np.array(d['sentence1_lemmatized'])
    sentence2_lemma_ds[:] = np.array(d['sentence2_lemmatized'])
    label_ds[:] = np.array(d['gold_label_int'])
    words_ds[:] = words

    ### h5py nonsense ###
    sentence1_ds_shapes[:] = np.array([np.array(x).shape for x in d['sentence1_tokenized']])
    sentence2_ds_shapes[:] = np.array([np.array(x).shape for x in d['sentence2_tokenized']])
    sentence1_lemma_ds_shapes[:] = np.array([np.array(x).shape for x in d['sentence1_lemmatized']])
    sentence2_lemma_ds_shapes[:] = np.array([np.array(x).shape for x in d['sentence2_lemmatized']])
    ds_shape_labels[:] = np.array(['sentence_len'], dtype='S20')

    def set_dims(ds, shapes):
        ds.dims.create_scale(shapes, 'shapes')
        ds.dims[0].attach_scale(shapes)
        ds.dims.create_scale(ds_shape_labels, 'shape_labels')
        ds.dims[0].attach_scale(ds_shape_labels)

    set_dims(sentence1_ds, sentence1_ds_shapes)
    set_dims(sentence1_lemma_ds, sentence1_lemma_ds_shapes)
    set_dims(sentence2_ds, sentence2_ds_shapes)
    set_dims(sentence2_lemma_ds, sentence2_lemma_ds_shapes)
    ### h5py nonsense ###

    print((len(d)))

    dst.attrs['split'] = H5PYDataset.create_split_array({
        'all': {
            'sentence1': (0, len(d)),
            'sentence2': (0, len(d)),
            'sentence1_lemmatized': (0, len(d)),
            'sentence2_lemmatized': (0, len(d)),
            'label': (0, len(d)),
            'text': (0, len(words))
        }
    })
    dst.close()


def breaking_nli_to_h5py_dataset(snli_path, dst_path):
    with open(snli_path) as file:
        dset = [eval(line.rstrip('\n')) for line in file]


    d = pd.DataFrame()
    d['sentence1_tokenized'] = [TreebankWordTokenizer().tokenize(dset[i]['sentence1']) for i in range(len(dset))]
    d['sentence2_tokenized'] = [TreebankWordTokenizer().tokenize(dset[i]['sentence2']) for i in range(len(dset))]
    d['gold_label'] = [dset[i]['gold_label'] for i in range(len(dset))]
    d['gold_label_int'] = [SNLI_LABEL2INT[x] for x in d['gold_label']]
    # Get all words
    sentences = d['sentence1_tokenized'] + d['sentence2_tokenized']
    words = np.array([w for s in tqdm.tqdm(sentences, total=len(sentences)) for w in s], dtype='S20')
    sentences = [] # For safety
    logging.info("Found {} words".format(len(words)))


    # Pack (I hate writing this h5py code)
    dtype = h5py.special_dtype(vlen='S20')
    dst = h5py.File(dst_path, "w")
    sentence1_ds = dst.create_dataset('sentence1', (len(d['sentence1_tokenized']),), dtype=dtype)
    sentence2_ds = dst.create_dataset('sentence2', (len(d['sentence1_tokenized']),), dtype=dtype)
    label_ds = dst.create_dataset('label', (len(d['sentence1_tokenized']),), dtype=np.int32)
    words_ds = dst.create_dataset('text', (len(words),), dtype='S20')

    ### h5py nonsense ###
    sentence1_ds_shapes = dst.create_dataset('sentence1_ds_shapes', (len(d['sentence1_tokenized']), 1), dtype=("int"))
    sentence2_ds_shapes = dst.create_dataset('sentence2_ds_shapes', (len(d['sentence1_tokenized']), 1), dtype=("int"))
    ds_shape_labels = dst.create_dataset('ds_shape_labels', (1,), dtype="S20")
    ### h5py nonsense ###
    sentence1_ds[:] = np.array(d['sentence1_tokenized'])
    sentence2_ds[:] = np.array(d['sentence2_tokenized'])

    label_ds[:] = np.array(d['gold_label_int'])
    words_ds[:] = words

    ### h5py nonsense ###
    sentence1_ds_shapes[:] = np.array([np.array(x).shape for x in d['sentence1_tokenized']])
    sentence2_ds_shapes[:] = np.array([np.array(x).shape for x in d['sentence2_tokenized']])
    ds_shape_labels[:] = np.array(['sentence_len'], dtype='S20')

    sentence1_ds.dims.create_scale(sentence1_ds_shapes, 'shapes')
    sentence1_ds.dims[0].attach_scale(sentence1_ds_shapes)
    sentence1_ds.dims.create_scale(ds_shape_labels, 'shape_labels')
    sentence1_ds.dims[0].attach_scale(ds_shape_labels)

    sentence2_ds.dims.create_scale(sentence2_ds_shapes, 'shapes')
    sentence2_ds.dims[0].attach_scale(sentence2_ds_shapes)
    sentence2_ds.dims.create_scale(ds_shape_labels, 'shape_labels')
    sentence2_ds.dims[0].attach_scale(ds_shape_labels)
    ### h5py nonsense ###

    print((len(d['sentence1_tokenized'])))

    dst.attrs['split'] = H5PYDataset.create_split_array({
        'all': {
            'sentence1': (0, len(d['sentence1_tokenized'])),
            'sentence2': (0, len(d['sentence1_tokenized'])),
            'label': (0, len(d['sentence1_tokenized'])),
            'text': (0, len(words))
        }
    })
    dst.close()