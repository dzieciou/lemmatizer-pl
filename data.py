# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import Levenshtein
import numpy
import os
import pickle

from collections import defaultdict
import gensim

from tagset import *
from constants import *
from concraft_utils import train_concraft_model, tag_data_with_concraft

WORD2VEC_MODELS_DIR = '/home/kkrasnowska/SCWAD/models/axw/orth'

def read_chunks(datafile, settings, MAX=None):
    chunks = []
    with codecs.open(datafile, encoding='utf-8', mode='r') as f:
        chunk = []
        l = f.readline()
        while l:
            l = l.strip('\n')
            if l and l not in ('EOS', 'EOP'):
                w, morf_in, morf_out, lab, _, _ = l.split('\t')
                chunk.append((w, morf_in, morf_out, lab))
            elif (l == 'EOP' and settings.FEED == 'paragraphs') or (l == 'EOS' and settings.FEED == SENT_FEED):
                chunks.append(chunk)
                chunk = []
                if len(chunks) % 1000 == 0:
                    print('    ...', len(chunks), 'chunks read')
                if MAX is not None and len(chunks) == MAX:
                    break
            l = f.readline()
    return chunks

class Indices(object):
    
    def __init__(self, task_id):
        self.TASK_ID = task_id
        self.IN_TAGS = []
        self.IN_TAG2INDEX = None
        self.IN_NUMTAGS = None
        self.OUT_TAGS = []
        self.OUT_TAG2INDEX = None
        self.OUT_NUMTAGS = None
        self.CHAR2INDEX = None
        self.NUMCHARS = None
        self.NUM_SUFFIXES = None
        self.SUFFIX2INDEX = None
        self.WORD2VEC = None
        self.SUFFIX_INDEX = None
        self.CONCRAFT_MODEL = None
    
    def createTagIndices(self, settings):
        for grp_no, tags in enumerate(TAG_GROUPS):
            if grp_no in settings.INPUT_TAGS:
                self.IN_TAGS += tags
            if grp_no in settings.OUTPUT_TAGS:
                self.OUT_TAGS += tags
        self.IN_TAG2INDEX = { t : i for i, t in enumerate(self.IN_TAGS) }
        self.IN_NUMTAGS = len(self.IN_TAGS)
        self.OUT_TAG2INDEX = { t : i for i, t in enumerate(self.OUT_TAGS) }
        self.OUT_NUMTAGS = len(self.OUT_TAGS)
    
    def createCharIndices(self, data, settings):
        CHARS = {'<UNK>'}
        for chunk in data:
            for item in chunk:
                wordform = item[0]
                CHARS.update(wordform)
        CHARS = sorted(CHARS)
        self.CHAR2INDEX = { c : i for i, c in enumerate(CHARS) }
        self.NUMCHARS = len(CHARS)
    
    def createSuffixEmbeddingIndices(self, data, settings):
        if SUFFIX_EMBED in settings.VECTOR_REPRS:
            suffs = set()
            for chunk in data:
                for item in chunk:
                    wordform = item[0]
                    suffs.add(wordform[-settings.SUFFIX_LEN:])
            suffs = sorted(suffs)
            self.NUM_SUFFIXES = len(suffs)
            self.SUFFIX2INDEX = { s : i + 1 for i, s in enumerate(suffs) }
    
    def createW2VIndices(self, settings):
        if W2V in settings.VECTOR_REPRS:
            WORD2VEC_MODEL = os.path.join(WORD2VEC_MODELS_DIR, 'w2v_allwiki_nkjpfull_%d.model' % settings.WORD2VEC_DIM)
            print('loading WORD2VEC model... ')
            self.WORD2VEC = gensim.models.word2vec.Word2Vec.load(WORD2VEC_MODEL)
            print('building suffix index... ')
            self.SUFFIX_INDEX = defaultdict(set)
            for w in self.WORD2VEC.vocab:
                for i in range(1, len(w)):
                    self.SUFFIX_INDEX[w[-i:]].add(w)
            print('done.')
    
    def createConcraftModel(self, data, settings):
        if CONCRAFT in settings.VECTOR_REPRS:
            self.CONCRAFT_MODEL = train_concraft_model(data, self.TASK_ID, settings)
    
    @staticmethod
    def create_indices(data, settings, task_id):
        indices = Indices(task_id)
        indices.createTagIndices(settings)
        indices.createCharIndices(data, settings)
        indices.createSuffixEmbeddingIndices(data, settings)
        indices.createW2VIndices(settings)
        indices.createConcraftModel(data, settings)
        return indices
    
    @staticmethod
    def load_indices(path, settings, task_id):
        indices = Indices(task_id)
        with open(path, 'r') as f:
            dct = pickle.load(f)
        for attr, val in dct.items():
            setattr(indices, attr, val)
        indices.createW2VIndices(settings)
        return indices
     
    def save(self, path):
        dct = { attr : val for attr, val in self.__dict__.items() if attr not in ('WORD2VEC', 'SUFFIX_INDEX') }
        with open(path, 'w') as f:
            pickle.dump(dct, f)


def morf2vector(morf, indices, settings, input_tag=True):
    #print(morf)
    tags = set()
    for morftag in morf.split('|'):
        tags.update(morftag.split(':'))
    vec = numpy.zeros((indices.IN_NUMTAGS))
    offset = 0
    relevant_grps = settings.INPUT_TAGS if input_tag else settings.OUTPUT_TAGS
    index = indices.IN_TAG2INDEX if input_tag else indices.OUT_TAG2INDEX
    for grp_no in relevant_grps:
        grp = TAG_GROUPS[grp_no]
        ts = tags.intersection(grp)
        if ts:
            for t in ts:
                vec[index[t]] = 1
        else:
            # mark the NOXYZ tag
            vec[index[grp[-1]]] = 1
    #print(vec)
    return vec

def splitMorfVector(vector, settings):
    vecs = []
    offset = 0
    for grp_no in settings.OUTPUT_TAGS:
        l = len(TAG_GROUPS[grp_no])
        vecs.append(vector[offset:(offset + l)])
        offset += l
    return vecs

#missing = set()
def word2EmbeddingVec(word, indices, settings):
    if word in indices.WORD2VEC:
        return indices.WORD2VEC[word]
    #if w not in missing:
    #    missing.add(word)
    # return the mean of vectors for most orthographically similar words
    for i in reversed(range(1, len(word))):
        suff = word[-i:]
        if suff in indices.SUFFIX_INDEX:
            alternatives = sorted((Levenshtein.distance(unicode(w), word), w) for w in indices.SUFFIX_INDEX[suff])
            min_dist = alternatives[0][0]
            alternatives = [indices.WORD2VEC[w] for d, w in alternatives if d == min_dist]
            return numpy.mean(alternatives, axis=0)
    #if word not in missing:
    #    missing.add(word)
    return numpy.random.uniform(-1, 1, settings.WORD2VEC_DIM)

def data2tensors(data, indices, settings, data_id=None):
    
    MAX_WORDS = 0
    for chunk in data:
        MAX_WORDS = max(MAX_WORDS, len(chunk))
    
    X_s = [numpy.zeros((len(data), MAX_WORDS, indices.IN_NUMTAGS)), None, None, None, None, None, None]
    
    # add char LSTM input
    if CHAR_LSTM in settings.VECTOR_REPRS:
        X_s[1] = numpy.zeros((len(data), MAX_WORDS, settings.SUFFIX_LEN, indices.NUMCHARS))
        # auxiliary layer for masking
        X_s[2] = numpy.zeros((len(data), MAX_WORDS, 128))
    
    # add suffix input
    if SUFFIX in settings.VECTOR_REPRS:
        X_s[3] = numpy.zeros((len(data), MAX_WORDS, indices.NUMCHARS * settings.SUFFIX_LEN))
    
    # add suffix embedding input
    if SUFFIX_EMBED in settings.VECTOR_REPRS:
        X_s[4] = numpy.zeros((len(data), MAX_WORDS))
    
    # add embeddings input
    if W2V in settings.VECTOR_REPRS:
        X_s[5] = numpy.zeros((len(data), MAX_WORDS, settings.WORD2VEC_DIM))
    
    # add Concraft input: weight + tag
    if CONCRAFT in settings.VECTOR_REPRS:
        L = indices.IN_NUMTAGS + (1 if CONCRAFT_WEIGHTS in settings.VECTOR_REPRS else 0)
        X_s[6] = numpy.zeros((len(data), MAX_WORDS, L))
        # don’t tag the same data more than once if possible
        if data_id is not None:
            tags_path = os.path.join(os.path.dirname(indices.CONCRAFT_MODEL), '%s.tags' % data_id)
            if os.path.isfile(tags_path):
                with open(tags_path, 'r') as f:
                    concraft_tags = pickle.load(f)
            else:            
                concraft_tags = tag_data_with_concraft(data, indices.CONCRAFT_MODEL)
                with open(tags_path, 'w') as f:
                    pickle.dump(concraft_tags, f)
        else:
            concraft_tags = tag_data_with_concraft(data, indices.CONCRAFT_MODEL)
    
    Y_s = [numpy.zeros((len(data), MAX_WORDS, len(TAG_GROUPS[grp_no]))) for  grp_no in settings.INPUT_TAGS]
    for chunk_i, chunk in enumerate(data):
        for item_i, (word, morf_in, morf_out, lab) in enumerate(chunk):
            v_in = morf2vector(morf_in, indices, settings)
            X_s[0][chunk_i][item_i] = v_in
            for char_i, char in enumerate(reversed(word)):
                if char_i < settings.SUFFIX_LEN:
                    if char not in indices.CHAR2INDEX:
                        char = '<UNK>'
                    # char LSTM input & auxiliary mask
                    if X_s[1] is not None:
                        X_s[1][chunk_i][item_i][char_i][indices.CHAR2INDEX[char]] = 1
                    # suffix input
                    if X_s[3] is not None:
                        X_s[3][chunk_i][item_i][char_i * indices.NUMCHARS + indices.CHAR2INDEX[char]] = 1
                # auxiliary mask
                if X_s[2] is not None:
                    X_s[2][chunk_i][item_i] = numpy.ones((128,))
            # suffix embedding input
            if X_s[4] is not None:
                X_s[4][chunk_i][item_i] = indices.SUFFIX2INDEX.get(word[-settings.SUFFIX_LEN:], indices.NUM_SUFFIXES + 1)
            # embeddings input
            if X_s[5] is not None:
                X_s[5][chunk_i][item_i] = word2EmbeddingVec(word, indices, settings)
            # Concraft input
            if X_s[6] is not None:
                c_tag, c_weight = concraft_tags[chunk_i][item_i]
                X_s[6][chunk_i][item_i][:indices.IN_NUMTAGS] = morf2vector(c_tag, indices, settings)
                if CONCRAFT_WEIGHTS in settings.VECTOR_REPRS:
                    X_s[6][chunk_i][item_i][-1] = c_weight
            for i, v_out in enumerate(splitMorfVector(morf2vector(morf_out, indices, settings, input_tag=False), settings)):
                Y_s[i][chunk_i][item_i] = v_out
    
    return [X for X in X_s if X is not None], Y_s
