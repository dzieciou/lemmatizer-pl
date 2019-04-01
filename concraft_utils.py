# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import os
import subprocess

from collections import defaultdict

from tagset import *
from constants import *

CONCRAFT_BIN = 'concraft/concraft-dag2-pl'
DATA_DIR = 'concraft-data'

def create_dag_file(data, dag_path):
    with codecs.open(dag_path, encoding='utf-8', mode='w') as dag:
        for chunk in data:
            for i, (form, morf_in, tag, _) in enumerate(chunk):
                match = 0
                for candidate_tag in morf_in.split('|'):
                    w = '0.000'
                    if candidate_tag == tag:
                        match += 1
                        w = '1.000'
                    # print form as lemma – doesn’t matter here
                    print('\t'.join((str(i + 1), str(i + 2), form, form, candidate_tag, w)), file=dag)
                if morf_in == 'ign':
                    match += 1
                    print('\t'.join((str(i + 1), str(i + 2), form, form, tag, '1.000')), file=dag)
                assert (match == 1)
            print(file=dag)
    return dag_path

def get_tagset_file(model_id, settings):
    if settings.TAGSET == NKJP:
        return os.path.join(DATA_DIR, 'nkjp-tagset-mod-2.cfg')
    # TODO add ‘Morfeusz-Rewolucje’ tagset file for Concraft
    raise RuntimeError


def train_concraft_model(data, model_id, settings):
    MODEL_DIR = os.path.join(DATA_DIR, model_id)
    subprocess.call('mkdir %s' % MODEL_DIR, shell=True)
    model_path = os.path.join(MODEL_DIR, model_id + '.gz')
    if not os.path.isfile(model_path):
        dag_path = create_dag_file(data, os.path.join(MODEL_DIR, model_id + '_train.dag'))
        tagset_path = get_tagset_file(model_id, settings)
        subprocess.call('%s train %s -e %s --tagsetpath=%s --batchsize=10 -o %s +RTS -N12 -A256M -s' % (CONCRAFT_BIN, dag_path, dag_path, tagset_path, model_path), shell=True)
    return model_path

def extract_tags(tagged_path):
    chunk_tags = []
    curr_chunk = defaultdict(set)
    with codecs.open(tagged_path, encoding='utf-8', mode='r') as dag:
        for l in dag:
            l = l.strip('\n')
            if not l:
                chunk_tags.append([])
                for i, tags in sorted(curr_chunk.items()):
                    assert (len(tags) == 1)
                    chunk_tags[-1].append(tags.pop())
                curr_chunk = defaultdict(set)
            else:
                elems = l.split('\t')
                # workaround for ‘ign’ bug
                if len(elems) == 6:
                    elems.append('')
                i, j, _, _, tag, weight, disamb = elems
                if disamb:
                    curr_chunk[int(i)].add((tag, float(weight)))
        # tagged file doesn’t have the final empty line
        chunk_tags.append([])
        for i, tags in sorted(curr_chunk.items()):
            assert (len(tags) == 1)
            chunk_tags[-1].append(tags.pop())
    return chunk_tags

def tag_data_with_concraft(data, model_path):
    input_path = create_dag_file(data, model_path.replace('.gz', '_input.dag'))
    tagged_path = model_path.replace('.gz', '_tagged.dag')
    subprocess.call('%s tag %s < %s > %s' % (CONCRAFT_BIN, model_path, input_path, tagged_path), shell=True)
    tags = extract_tags(tagged_path)
    assert (len(data) == len(tags))
    for chunk_i, chunk in enumerate(data):
        tgs = tags[chunk_i]
        assert (len(chunk) == len(tgs))
    return tags
