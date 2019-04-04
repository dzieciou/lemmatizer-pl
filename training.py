#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import argparse
import datetime
import numpy
import os
import pickle
import random
import subprocess
import time

from multiprocessing import Pool

from keras.models import load_model
from keras.utils import plot_model

from constants import *
from data import read_chunks, Indices, data2tensors
from network import build_model
from evaluation import evaluate_model
from settings import Settings

SETTINGS = Settings()
HASH = str(hash(SETTINGS))

def train_model(data, settings, task_id):
    t1 = datetime.datetime.now()
    indices = Indices.create_indices(data, settings, task_id)
    data_sorted = sorted(data, key=len)
    model = build_model(indices, settings)
    batch_indices = list(range(len(data_sorted) // settings.BUCKET_SIZE + 1))
    for EPOCH in range(settings.EPOCHS):
        random.shuffle(batch_indices)
        for b, i in enumerate(batch_indices):
            print('EPOCH:', EPOCH, 'BUCKET:', b + 1)
            index = i * settings.BUCKET_SIZE
            X_trains, Y_trains = data2tensors(data_sorted[index:(index + settings.BUCKET_SIZE)], indices, settings, data_id=('%s_%d' % (task_id, i)))
            for x_i in range(len(X_trains)):
                print(X_trains[x_i][0][0])
            model.fit(X_trains, Y_trains, epochs=1, batch_size=128, verbose=2)
            del X_trains
            del Y_trains
    t2 = datetime.datetime.now()
    print('TOTAL TRAINING TIME:', t2 - t1)
    return model, indices

def train_and_evaluate(args):
    train, test, settings, task_id = args
    numpy.random.seed(settings.SEED)
    random.seed(settings.SEED)
    # in case there is no wordembeddings directory
    subprocess.call('mkdir wordembeddings', shell=True)
    # load or train&save the model
    MODEL_DIR = os.path.join('wordembeddings', HASH)
    MODEL_PATH = os.path.join(MODEL_DIR, '%s_el-modelito.h5' % task_id)
    GRAPH_PATH = os.path.join(MODEL_DIR, '%s_el-modelito.png' % task_id)
    SETTINGS_PATH = os.path.join(MODEL_DIR, 'SETTINGS.pickle')
    INDICES_PATH = os.path.join(MODEL_DIR, '%s_indices.pickle' % task_id)
    if os.path.isfile(MODEL_PATH):
        print('reading model from file:', MODEL_PATH)
        model = load_model(MODEL_PATH)
        indices = Indices.load_indices(INDICES_PATH, settings, task_id)
    else:
        model, indices = train_model(train, settings, task_id)
        subprocess.call('mkdir %s' % MODEL_DIR, shell=True)
        model.save(MODEL_PATH)
        plot_model(model, show_shapes=True, to_file=GRAPH_PATH)
        indices.save(INDICES_PATH)
        with open(SETTINGS_PATH, 'wb') as f:
            pickle.dump(settings, f)
    # evaluate
    scores_train = evaluate_model(model, train, indices, settings, task_id + '_train_eval')
    scores_test = evaluate_model(model, test, indices, settings, task_id + '_test_eval')
    del model
    return scores_test, scores_train

def print_scores(scores):
    t_x, t_y = 0, 0
    for et in scores[0].keys():
        x = sum(s[et][0] for s in scores)
        y = sum(s[et][1] for s in scores)
        print('%s: %d' % (et, y))
        if y:
            print('   %.4f' % (100 * x / y))
        t_x += x
        t_y += y
    print('\nTOTAL: %d' % t_y)
    if t_y:
        print('   %.4f' % (100 * t_x / t_y))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=10,
        help='number of folds for cross-validation; set to <2 to train single model on whole training data; default: 10')
    parser.add_argument('--nproc', type=int, default=5,
        help='number of processes to be used in parallel; default: 5')
    args = parser.parse_args()
    
    FOLDS = args.folds
    CROSS_VALID = (FOLDS > 1)
    NPROC = args.nproc
    
    train_data = []
    to_read = SETTINGS.MAX_TRAIN_CHUNKS
    for trainfile in SETTINGS.TRAIN:
        if to_read == 0:
            break
        print('trainfile:', trainfile)
        train_data += read_chunks(trainfile, SETTINGS, MAX=to_read)
        to_read = SETTINGS.MAX_TRAIN_CHUNKS - len(train_data)
    print('\n%d training chunks' % len(train_data))
    
    FOLD_SIZE = len(train_data) // FOLDS
    train_data = train_data[:(FOLDS * FOLD_SIZE)]
    print('\n%d training chunks used' % len(train_data))
    tasks = []
    if CROSS_VALID:
        for fold in range(FOLDS):
            lo = fold * FOLD_SIZE
            hi = lo + FOLD_SIZE
            print(hi, lo)
            train = train_data[:lo] + train_data[hi:]
            test = train_data[lo:hi]
            print('TRAIN:', len(train), 'TEST:', len(test))
            tasks.append((train, test, SETTINGS, '%s_%d_%d' % (SETTINGS.DATA_ID, lo, hi)))
    else:
        tasks.append((train_data, [], SETTINGS, '%s_ALL_%d' % (SETTINGS.DATA_ID, len(train_data))))
    
    if NPROC > 1:
        pool = Pool(processes=NPROC)
        SCORES = pool.map(train_and_evaluate, tasks)
    else:
        SCORES = map(train_and_evaluate, tasks)
    scores_test, scores_train = zip(*SCORES)
    print('RESULTS ON *TRAIN* DATA:')
    print_scores(scores_train)
    print('\nRESULTS ON *TEST* DATA:')
    print_scores(scores_test)

if __name__ == '__main__':
    main()
