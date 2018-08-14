#!/usr/bin/env python

'''
Downloads SNLI dataset as zip file and unpacks it.
'''

import argparse
import jsonlines
import os

def build_sequence(filepath, dst_dir):
    dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}
    filename = os.path.basename(filepath)
    print(filename)
    len_p = []
    len_h = []
    with open(filepath) as f, \
         open(os.path.join(dst_dir, 'premise_%s'%filename), 'w') as f1, \
         open(os.path.join(dst_dir, 'hypothesis_%s'%filename), 'w') as f2,  \
         open(os.path.join(dst_dir, 'label_%s'%filename), 'w') as f3:
        next(f) # skip the header row
        for line in f:
            sents = line.strip().split('\t')
            if sents[0] is '-':
                continue

            words_in = sents[1].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(',')')]
            f1.write(' '.join(words_in) + '\n')
            len_p.append(len(words_in))

            words_in = sents[2].strip().split(' ')
            words_in = [x for x in words_in if x not in ('(',')')]
            f2.write(' '.join(words_in) + '\n')
            len_h.append(len(words_in))

            f3.write(dic[sents[0]] + '\n')


def build_sequence_breaking(filepath, dst_dir):
    dic_label = {
        'entailment': '0',
        'neutral': '1',
        'contradiction': '2'
    }

    filename = 'breaking'
    with open(filepath) as f, \
         open(os.path.join(dst_dir, 'premise_%s.txt' % filename), 'w') as f1, \
         open(os.path.join(dst_dir, 'hypothesis_%s.txt' % filename), 'w') as f2, \
         open(os.path.join(dst_dir, 'label_%s.txt' % filename), 'w') as f3, \
         jsonlines.open(filepath) as reader:
        for obj in reader:
            f1.write(obj['sentence1'] + '\n')
            f2.write(obj['sentence2'] + '\n')
            f3.write(dic_label[obj['gold_label']] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == 'snli':
        build_sequence(os.path.join(args.input_dir, 'snli_1.0_dev.txt'), args.output_dir)
        build_sequence(os.path.join(args.input_dir, 'snli_1.0_test.txt'), args.output_dir)
        build_sequence(os.path.join(args.input_dir, 'snli_1.0_train.txt'), args.output_dir)
    elif args.type == 'breaking':
        build_sequence_breaking(os.path.join(args.input_dir, 'test_breaking_nli.jsonl'), args.output_dir)
