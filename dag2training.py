#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import codecs
import os

from collections import defaultdict

def append_to_tab(paragraph, outfile):
    segments = defaultdict(lambda: ['', set()])
    in_interps = defaultdict(set)
    out_interps = defaultdict(set)
    labels = defaultdict(str)
    end_of_sent = defaultdict(str)
    for l in paragraph:
        i, j, orth, base, tag, _, _, w, label, eos = l.split('\t')
        i, j = int(i), int(j)
        w = float(w)
        if not label:
            # don’t input INTERP nor IGN tags (inconsistent with Morfeusz)
            # – force the tagger to guess!
            in_interps[(i, j)].add(tag)
        if w > 0:
            out_interps[(i, j)].add(tag)
        segments[(i, j)][0] = orth
        if label:
            labels[(i, j)] = label
        if eos:
            end_of_sent[(i, j)] = eos
    for (i, j), (orth, tags) in sorted(segments.items()):
        if not out_interps[(i, j)]:
            # don’t print out segments from discarded segmentation variants
            # TODO make the tagger handle segmentation as well?
            continue
        morf_in = '|'.join(sorted(in_interps[(i, j)]))
        morf_out = '|'.join(sorted(out_interps[(i, j)]))
        row = (orth, morf_in, morf_out, labels[(i, j)], '', '')
        print('\t'.join(row), file=outfile)
        if end_of_sent[(i, j)]:
            print('EOS', file=outfile)
    print('EOP', file=outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    
    output_file = args.output
    if output_file is None:
        pth, _ = os.path.splitext(path)
        output_file = pth + '.tab'
    
    # split paragraphs into folds and convert morphology to vectors
    par_count = 0
    par = []
    with codecs.open(args.input_file, encoding='utf=8', mode='r') as dag:
        with codecs.open(output_file, encoding='utf=8', mode='w') as tab:
            l = dag.readline()
            while l:
                l = l.strip('\n')
                if l:
                    par.append(l)
                else:
                    append_to_tab(par, tab)
                    par = []
                    par_count += 1
                    if par_count % 500 == 0:
                        print('...', par_count, 'paragraphs processed')
                l = dag.readline()
    
if __name__ == '__main__':
    main()
