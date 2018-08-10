#!/usr/bin/env python

'''
Downloads SNLI dataset as zip file and unpacks it.
'''

import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('snli_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    build_sequence(os.path.join(args.snli_dir, 'snli_1.0_dev.txt'), args.output_dir)
    build_sequence(os.path.join(args.snli_dir, 'snli_1.0_test.txt'), args.output_dir)
    build_sequence(os.path.join(args.snli_dir, 'snli_1.0_train.txt'), args.output_dir)
