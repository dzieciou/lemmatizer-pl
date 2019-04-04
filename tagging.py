#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import codecs
import os
import pickle
from collections import defaultdict
from xml.dom.minidom import parse

import morfeusz2
import numpy
from keras.models import load_model

from PolEval.prepare_data import extractSents
from data import Indices
from evaluation import evaluate_model


#===============================================================================

def readTXTsentences(path):
    sentences = []
    with codecs.open(path, encoding='utf=8', mode='r') as f:
        for l in f:
            sentences.append(l.strip('\n'))
    return sentences

def txt2data(path, args):
    dict_path, dict_name = os.path.split(args.morfeusz_dict)
    morfeusz = morfeusz2.Morfeusz(dict_name=dict_name, dict_path=dict_path, expand_tags=True, generate=False)
    sentences = readTXTsentences(path)
    chunks = []
    for sentence in sentences:
        seg_orths, seg_tags = dict(), defaultdict(set)
        for (i, j, (orth, _, tag, _, _)) in morfeusz.analyse(sentence):
            i, j = int(i), int(j)
            seg_orths[(i, j)] = orth
            if tag not in ('dig', 'romandig'):
                seg_tags[(i, j)].add(tag)
        chunks.append([])
        for indices in sorted(seg_orths.keys()):
            orth = seg_orths[indices]
            tags = set(seg_tags[indices])
            if not tags:
                # possible for dig/romandig
                tags = set(['ign'])
            tag = '_'
            label = 'IGN' if tags == set(['ign']) else ''
            chunks[-1].append((orth, '|'.join(tags), tag, label))
    return chunks

def outputTXT(data, chunk_tags, input_path, output_path):
    with codecs.open(output_path, encoding='utf-8', mode='w') as txt:
        for sent_i, sent in enumerate(data):
            tags = chunk_tags[sent_i]
            for tok_i, tok in enumerate(sent):
                orth = tok[0]
                tag = tags[tok_i]
                print(orth, tag, file=txt)
            print(file=txt)

def outputDAG(data, chunk_tags, input_path, output_path, args):
    dict_path, dict_name = os.path.split(args.morfeusz_dict)
    morfeusz = morfeusz2.Morfeusz(expand_tags=True, generate=False)
    sentences = readTXTsentences(input_path)
    with codecs.open(output_path, encoding='utf-8', mode='w') as dag:
        for sent_i, sentence in enumerate(sentences):
            tags = chunk_tags[sent_i]
            seg_interps = defaultdict(list)
            for (i, j, interp) in morfeusz.analyse(sentence):
                i, j = int(i), int(j)
                seg_interps[(i, j)].append(interp)
            for tok_i, (indices, interps) in enumerate(sorted(seg_interps.items())):
                i, j = indices
                disamb_tag = tags[tok_i]
                match = False
                for orth, base, tag, posp, kwal in interps:
                    assert (orth == data[sent_i][tok_i][0])
                    disamb = ''
                    if tag == disamb_tag:
                        disamb, match = 'disamb', True
                    if disamb or args.full_output:
                        print('\t'.join(map(str, (i, j, orth, base, tag, '|'.join(posp), '|'.join(kwal), disamb))), file=dag)
                if not match:
                    orth = interps[0][0]
                    base = orth
                    print('\t'.join(map(str, (i, j, orth, base, disamb_tag, '', '', 'disamb'))), file=dag)
            print(file=dag)
    
#===============================================================================

def XCES_analysed2data(path):
    analysed = extractSents(path)
    chunks = []
    for an in analysed:
        chunks.append([])
        for tok in an:
            orth = tok[0]
            tags = set(list(zip(*tok[1]))[2])
            tag = '_'
            if 'ign' in tags:
                tags = set(['ign'])
            label = 'IGN' if tags == set(['ign']) else ''
            chunks[-1].append((orth, '|'.join(tags), tag, label))
    return chunks

def outputXCES(data, chunk_tags, input_path, output_path, args):
    # prepare the result XML
    DOM = parse(input_path)
    sents = filter(lambda e: e.getAttribute('type') == 's', DOM.getElementsByTagName('chunk'))
    for sent_i, sent in enumerate(sents):
        tags = chunk_tags[sent_i]
        for tok_i, tok in enumerate(sent.getElementsByTagName('tok')):
            orth = tok.getElementsByTagName('orth')[0].childNodes[0].data
            assert (orth == data[sent_i][tok_i][0])
            # find a matching lemma and reduce number of lex children to one
            lemmata = set()
            for i, lex in enumerate(tok.getElementsByTagName('lex')):
                lex_tag = lex.getElementsByTagName('ctag')[0].childNodes[0].data
                if lex_tag == tags[tok_i]:
                    lemmata.add(lex.getElementsByTagName('base')[0].childNodes[0].data)
                if i > 0:
                    tok.removeChild(lex)
            # default lemma: orth
            if not lemmata:
                lemmata = set([orth])
            # modify the only remaining lex node
            the_lex = tok.getElementsByTagName('lex')[0]
            the_lex.getElementsByTagName('base')[0].childNodes[0].data = sorted(lemmata)[0]
            the_lex.getElementsByTagName('ctag')[0].childNodes[0].data = tags[tok_i]
            the_lex.setAttribute('disamb', '1')
    # print out the result XML
    with codecs.open(output_path, encoding='utf-8', mode='w') as xml:
        for l in DOM.toxml().split('\n'):
            if l.strip():
                print(l, file=xml)

#===============================================================================

def makeTaggedPath(path):
    pth, ext = os.path.splitext(path)
    return pth + '.tagged' + ext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('input_format', choices=['xces', 'dag', 'txt'])
    parser.add_argument('model', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-b', '--batch', action='store_true', help='with this flag, <input_file> is expected to contain a list of paths (one per line) of files in <input_format> to be processed')
    parser.add_argument('--full_output', action='store_true', help='output will all interpretations')
    parser.add_argument('--morfeusz_dict', type=str, help='required when <input_format> is ‘txt’')
    args = parser.parse_args()
    
    if args.input_format not in ('xces', 'txt'):
        print(u'I’m terribly sorry, but the ‘%s’ input format is currently not supported!\n' % args.input_format)
        return 1
    if args.input_format == 'txt' and args.morfeusz_dict is None:
        print(u'‘txt’ input format requires --morfeusz_dict to be specified!\n')
        return 1
    
    if args.input_format == 'xces' and args.full_output:
        print(u'WARNING: --full_output option will have no effect for ‘xces’ input format!\n')
    
    # files to be tagged
    input_files = []
    if args.batch:
        with codecs.open(args.input_file, encoding='utf-8', mode='r') as f:
            for l in f:
                input_path = l.strip('\n')
                input_files.append((input_path, makeTaggedPath(input_path)))
    else:
        input_path = args.input_file
        output_path = args.output
        if output_path is None:
            output_path = makeTaggedPath(input_path)
        input_files.append((input_path, output_path))
    
    # read the model
    MODEL_PATH = args.model
    MODEL_DIR = os.path.dirname(MODEL_PATH)
    INDICES_PATH = MODEL_PATH.replace('el-modelito.h5', 'indices.pickle')
    print('reading model from file:', MODEL_PATH)
    model = load_model(MODEL_PATH)
    with open(os.path.join(MODEL_DIR, 'SETTINGS.pickle'), 'rb') as f:
        settings = pickle.load(f)
    numpy.random.seed(settings.SEED)
    task_id = os.path.basename(MODEL_PATH).replace('_el-modelito.h5', '')
    print(task_id)
    indices = Indices.load_indices(INDICES_PATH, settings, task_id)
    
    # tag the input
    for input_path, output_path in input_files:
        print('\n\nTAGGING:\n    %s' % input_path)
        if args.input_format == 'xces':
            data = XCES_analysed2data(input_path)
        elif args.input_format == 'txt':
            data = txt2data(input_path, args)
        _, chunk_tags = evaluate_model(model, data, indices, settings, 'foobar', eval_mode=False, return_tags=True)
        if args.input_format == 'xces':
            outputXCES(data, chunk_tags, input_path, output_path, args)
        elif args.input_format == 'txt':
            outputDAG(data, chunk_tags, input_path, output_path, args)
        print('RESULT SAVED TO:\n    %s' % output_path)
    
if __name__ == '__main__':
    main()
