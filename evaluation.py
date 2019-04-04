#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import datetime
import Levenshtein
import numpy
import os
import pickle

from keras.models import load_model

from collections import Counter, defaultdict

from tagset import *
from data import read_chunks, Indices, data2tensors

available_char = 33
TAG2CODE = {}
CODE2TAG = {}
for tag_grp in TAG_GROUPS:
    first_char = chr(available_char)
    available_char += 1
    for tag in tag_grp:
        if not tag.startswith('NO'):
            code = first_char + chr(available_char)
            TAG2CODE[tag] = code
            CODE2TAG[code] = tag
            available_char += 1

def tag2code(tag):
    return ''.join(TAG2CODE[t] for t in tag.split(':'))

def code2tag(code):
    return ':'.join(CODE2TAG[code[i:(i + 2)]] for i in range(0, len(code), 2))

EXAMPLE_TYPES = ('STANDARD', 'TRIVIAL', 'INTERP', 'IGN', 'SEG')

def evaluate_model(model, data, indices, settings, data_id, eval_mode=True, return_tags=False):
    time1 = datetime.datetime.now()
    mistakes = Counter()
    scores = { t : [0, 0] for t in EXAMPLE_TYPES }
    
    # index chunks by length & remember their original order
    buckets = defaultdict(list)
    buckets_index = dict()
    buckets_tags = defaultdict(lambda: defaultdict(list))
    for chunk_i, chunk in enumerate(data):
        L = len(chunk)
        buckets[L].append(chunk)
        buckets_index[chunk_i] = (L, len(buckets[L]) - 1)
    
    for L, bucket in sorted(buckets.items()):
        #print('chunk len:', L, 'chunks:', len(bucket))
        X_tests, Y_tests = data2tensors(bucket, indices, settings, data_id=('%s-%d' % (data_id, L)))
        model_predictions = model.predict(X_tests)
        for chunk_i, chunk in enumerate(bucket):
            for item_i, item in enumerate(chunk):
                w, morf_in, correct_tag, lab = item
                predicted = [predictions[chunk_i][item_i] for predictions in model_predictions]
                tags = []
                for tag_i, tag_pred in enumerate(predicted):
                    tag = TAG_GROUPS[tag_i][numpy.argmax(tag_pred)]
                    if not tag.startswith('NO'):
                        tags.append(TAG_GROUPS[tag_i][numpy.argmax(tag_pred)])
                predicted_tag = ':'.join(tags)
                candidate_tags = set(filter(None, set(morf_in.split('|'))))
                # ign means selection from all possible valid tags
                if 'ign' in candidate_tags:
                    candidate_tags = VALID_TAGS
                
                if lab:
                    et = lab
                else:
                    et = 'TRIVIAL' if len(candidate_tags) == 1 else 'STANDARD'
                
                if candidate_tags and predicted_tag not in candidate_tags:
                    pred_tag_C = tag2code(predicted_tag)
                    cand_tags_C = set(map(tag2code, candidate_tags))
                    distances = [(Levenshtein.distance(t, pred_tag_C), t) for t in cand_tags_C]
                    corrected_tag = code2tag(sorted(distances)[0][1])
                else:
                    corrected_tag = predicted_tag
                
                if corrected_tag not in VALID_TAGS:
                    print('!!!', corrected_tag, candidate_tags)
                
                if eval_mode and et == 'TRIVIAL' and corrected_tag != correct_tag:
                    print(w, '***', et)
                    print('  ', '|'.join(candidate_tags))
                    print('   predicted:', predicted_tag)
                    print('   corrected:', corrected_tag)
                    print('   correct:', correct_tag)
                
                scores[et][0] += corrected_tag == correct_tag
                scores[et][1] += 1
                if et == 'STANDARD' and corrected_tag != correct_tag:
                    mistakes[(correct_tag, corrected_tag)] += 1
                
                if return_tags:
                    buckets_tags[L][chunk_i].append(corrected_tag)
                
        del X_tests
        del Y_tests
        del model_predictions
    
    if eval_mode:
        for (t1, t2), n in mistakes.most_common(n=10):
            print('%40s %40s %d' % (t1, t2, n))
    
    time2 = datetime.datetime.now()
    action = 'EVALUATION' if eval_mode else 'TAGGING'
    print('%s TIME:' % action, time2 - time1)
    
    if return_tags:
        chunk_tags = []
        for chunk_i, chunk in enumerate(data):
            i, j = buckets_index[chunk_i]
            tags = buckets_tags[i][j]
            assert (len(tags) == len(chunk))
            chunk_tags.append(tags)
        return scores, chunk_tags
    else:
        return scores
