#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
from collections import defaultdict
from collections import namedtuple
from xml.dom.minidom import parse

import morfeusz2
import numpy
from keras.models import load_model

from PolEval.prepare_data import extractSents
from data import Indices
from evaluation import evaluate_model

# ===============================================================================

Result = namedtuple('Result', ['i', 'j', 'interp', 'disamb'])


class Library(object):

    def __init__(self):
        self.tagger = Tagger()
        model_path = 'wordembeddings\\1895132012650666018\\PolEval_ALL_85660_el-modelito.h5'
        self.disambiguator = Disambiguator(model_path)
        self.printer = DagPrinter()


    def tag(self, sent):
        chunks = self.tagger.tag_sentence(sent)
        chunk_tags = self.disambiguator.disambiguate(chunks)
        return self.printer.get_results(sent, chunk_tags[0])


class DagPrinter(object):

    def __init__(self, dict_path=None, dict_name=None, full_output=False):
        self.morfeusz = morfeusz2.Morfeusz(expand_tags=True, generate=False)
        self.full_output = full_output

    def output_dag(self, tagged_data, sents_tags, input_path, output_path):

        sentences = list(get_sentences_from_txt(input_path))
        with open(output_path, encoding='utf-8', mode='w') as dag:
            for sent_i, sent in enumerate(sentences):
                sent_tags = sents_tags[sent_i]
                for i, j, interp, disamb in self.get_results(sent,
                                                             sent_tags,
                                                             tagged_data[
                                                                 sent_i]):
                    orth, base, tag, posp, kwal = interp
                    formatted = '\t'.join(map(str, (
                        i, j, orth, base, tag, '|'.join(posp), '|'.join(kwal),
                        disamb)))
                    print(formatted, file=dag)
                print(file=dag)

    def get_results(self, sent, sent_tags, tagged_sent=None):

        seg_interps = defaultdict(list)
        for (i, j, interp) in self.morfeusz.analyse(sent):
            i, j = int(i), int(j)
            seg_interps[(i, j)].append(interp)

        for tok_i, (indices, interps) in enumerate(sorted(seg_interps.items())):
            i, j = indices
            disamb_tag = sent_tags[tok_i]
            if self.full_output:
                for interp in interps:
                    orth, base, tag, posp, kwal = interp
                    if tagged_sent is not None:
                        assert orth == tagged_sent[tok_i][0]
                    disamb = 'disamb' if tag == disamb_tag else ''
                    yield Result(i, j, interp, disamb=disamb)
            else:
                matching_interp = self.find_matching_interp(interps, disamb_tag)
                if matching_interp:
                    yield Result(i, j, matching_interp, disamb='disamb')
                else:
                    interp = interps[0]
                    orth = interp[0]
                    base = orth
                    interp = orth, base, disamb_tag, [], []
                    yield Result(i, j, interp, disamb='disamb')

    def find_matching_interp(self, interps, disamb_tag):
        try:
            matching_interps = ((orth, base, tag, posp, kwal) for
                                orth, base, tag, posp, kwal in interps if
                                tag == disamb_tag)
            return next(matching_interps)
        except StopIteration:
            return None


class Tagger(object):

    def __init__(self, dict_path=None, dict_name=None):
        self.morfeusz = morfeusz2.Morfeusz(expand_tags=True, generate=False)

    def tag_sentence(self, sent):
        chunks = []
        seg_orths, seg_tags = {}, defaultdict(set)
        for (i, j, (orth, _, tag, _, _)) in self.morfeusz.analyse(sent):
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
            chunk = (orth, '|'.join(tags), tag, label)
            chunks[-1].append(chunk)
        return chunks


class Disambiguator(object):

    def __init__(self, model_path):
        print('reading model from file:', model_path)
        model_dir = os.path.dirname(model_path)
        indices_path = model_path.replace('el-modelito.h5', 'indices.pickle')
        self.model = load_model(model_path)
        with open(os.path.join(model_dir, 'SETTINGS.pickle'), 'rb') as f:
            self.settings = pickle.load(f)
        numpy.random.seed(self.settings.SEED)
        task_id = os.path.basename(model_path).replace('_el-modelito.h5', '')
        print(task_id)
        self.indices = Indices.load_indices(indices_path, self.settings,
                                            task_id)

    def disambiguate(self, data):
        _, chunk_tags = evaluate_model(self.model,
                                       data, self.indices,
                                       self.settings,
                                       'foobar',
                                       eval_mode=False,
                                       return_tags=True)
        return chunk_tags


def get_sentences_from_txt(path):
    with open(path, encoding='utf=8', mode='r') as f:
        for l in f:
            sentence = l.strip('\n')
            yield sentence


def output_txt(sents, sents_tags, output_path):
    with open(output_path, encoding='utf-8', mode='w') as txt:
        for sent_i, sent in enumerate(sents):
            sent_tags = sents_tags[sent_i]
            for orth, tag in get_tagged_tokens(sent, sent_tags):
                print(orth, tag, file=txt)
            print(file=txt)


def get_tagged_tokens(sent, sent_tags):
    for token_i, token in enumerate(sent):
        orth = token[0]
        tag = sent_tags[token_i]
        yield orth, tag


# ===============================================================================

def get_tagged_data_from_xces(path):
    analysed = extractSents(path)
    chunks = []
    for an in analysed:
        chunks.append([])
        for tok in an:
            orth = tok[0]
            tags = set(list(zip(*tok[1]))[2])
            if 'ign' in tags:
                tags = set(['ign'])
            label = 'IGN' if tags == set(['ign']) else ''
            tag = '_'
            chunk = (orth, '|'.join(tags), tag, label)
            chunks[-1].append(chunk)
    return chunks


def output_xces(data, chunk_tags, input_path, output_path):
    # prepare the result XML
    DOM = parse(input_path)
    sents = filter(lambda e: e.getAttribute('type') == 's',
                   DOM.getElementsByTagName('chunk'))
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
                    lemmata.add(
                        lex.getElementsByTagName('base')[0].childNodes[0].data)
                if i > 0:
                    tok.removeChild(lex)
            # default lemma: orth
            if not lemmata:
                lemmata = {[orth]}
            # modify the only remaining lex node
            the_lex = tok.getElementsByTagName('lex')[0]
            the_lex.getElementsByTagName('base')[0].childNodes[0].data = \
                sorted(lemmata)[0]
            the_lex.getElementsByTagName('ctag')[0].childNodes[0].data = tags[
                tok_i]
            the_lex.setAttribute('disamb', '1')
    # print out the result XML
    with open(output_path, encoding='utf-8', mode='w') as xml:
        for l in DOM.toxml().split('\n'):
            if l.strip():
                print(l, file=xml)


# ===============================================================================

def make_tagged_path(path):
    pth, ext = os.path.splitext(path)
    return pth + '.tagged' + ext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('input_format', choices=['xces', 'dag', 'txt'])
    parser.add_argument('model', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-b', '--batch', action='store_true',
                        help='with this flag, <input_file> is expected to contain a list of paths (one per line) of files in <input_format> to be processed')
    parser.add_argument('--full_output', action='store_true',
                        help='output will all interpretations')
    parser.add_argument('--morfeusz_dict', type=str,
                        help='required when <input_format> is ‘txt’')
    args = parser.parse_args()

    input_format = args.input_format
    if input_format not in ('xces', 'txt'):
        print(
            f'I’m terribly sorry, but the ‘{input_format}’ input format is currently not supported!\n')
        return 1
    if input_format == 'txt' and args.morfeusz_dict is None:
        print(u'‘txt’ input format requires --morfeusz_dict to be specified!\n')
        return 1

    if input_format == 'xces' and args.full_output:
        print(
            u'WARNING: --full_output option will have no effect for ‘xces’ input format!\n')

    return args


def main():
    args = parse_args()

    files = get_files_to_tag(args.input_file, args.output, args.batch)

    dict_path, dict_name = os.path.split(args.morfeusz_dict)
    tagger = Tagger(dict_path, dict_name)

    model_path = args.model
    disambiguator = Disambiguator(model_path)

    input_format = args.input_format
    for input_path, output_path in files:

        tagged_data = get_tagged_data(input_format, input_path, tagger)

        print('\n\nDISAMBIGUATING:\n    %s' % input_path)
        chunk_tags = disambiguator.disambiguate(tagged_data)

        if input_format == 'xces':
            output_xces(tagged_data, chunk_tags, input_path, output_path)
        elif input_format == 'txt':
            dag_printer = DagPrinter(dict_path, dict_name, args.full_output)
            dag_printer.output_dag(tagged_data, chunk_tags, input_path,
                                   output_path)

        print('RESULT SAVED TO:\n    %s' % output_path)


def get_tagged_data(input_format, input_path, tagger):
    if input_format == 'xces':
        return get_tagged_data_from_xces(input_path)
    elif input_format == 'txt':
        return tag_txt_file(input_path, tagger)
    raise ValueError(f'Unknown input format: {input_format}')


def tag_txt_file(input_path, tagger):
    print('\n\nTAGGING:\n    %s' % input_path)
    data = []
    for sentence in get_sentences_from_txt(input_path):
        data.extend(tagger.tag_sentence(sentence))
    return data


def get_files_to_tag(input_file, output=None, batch=False):
    files = []
    if batch:
        with open(input_file, encoding='utf-8', mode='r') as f:
            for l in f:
                input_path = l.strip('\n')
                files.append((input_path, make_tagged_path(input_path)))
    else:
        input_path = input_file
        output_path = output
        if output_path is None:
            output_path = make_tagged_path(input_path)
        files.append((input_path, output_path))
    return files


if __name__ == '__main__':
    main()
