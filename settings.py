# -*- coding: utf-8 -*-

from tagset import *
from constants import *

class Settings(object):
    
    def __init__(self):
        self.MODEL_TYPE = BI_LSTM
        self.HIDDEN_LAYERS = 2
        # LSTM is bidirectional, so this is actually doubled
        self.HIDDEN_DIM = 384
        self.EPOCHS = 20
        self.BUCKET_SIZE = 2048
        self.TAGSET = NKJP
        # include only specified tag positions in input vectors
        self.INPUT_TAGS = tuple(range(len(TAG_GROUPS)))
        # predict only specified tag positions
        self.OUTPUT_TAGS = tuple(range(len(TAG_GROUPS)))
        self.VECTOR_REPRS = tuple(sorted((W2V,)))
        self.SUFFIX_LEN = 4
        self.WORD2VEC_DIM = 300
        self.FEED = SENT_FEED
        self.DATA_ID = 'PolEval'
        self.TRAIN = ('PolEval/train.tab',)#('../folds/%d.tab' % i for i in xrange(7))
        self.MAX_TRAIN_CHUNKS = 85660
        self.SEED = 42
    
    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
