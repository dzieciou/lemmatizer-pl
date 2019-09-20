"""
Train MorphDisambiguator and save its model to file.
"""

import logging

from gensim.models import KeyedVectors
from tensorflow.python.util import deprecation

from eval import timing
from lemmatizer import nkjp
from lemmatizer.disamb import MorphDisambiguator
from xces import load_chunks_set

deprecation._PRINT_DEPRECATION_WARNINGS = False



def check_ctag(ctag, tagset):
    if not ctag in tagset.valid_ctags:
        raise ValueError(f'Invalid ctag: {ctag}')


def check_chunks(chunks, tagset):
    for chunk in chunks:
        for token in chunk.tokens:
            if not token.ctags is None:
                for ctag in token.ctags:
                    check_ctag(ctag, tagset)



def check(chunks_X, chunks_y, tagset):

    with timing('Checking train data against tagset'):
        check_chunks(chunks_X, tagset)
        check_chunks(chunks_y, tagset)

    # TODO Move this to checks for PosTagger or MorphAnalyzer
    #      disambiguator is using only tagset, not dictionary
    # with timing('Checking dictionary against tagset'):
    #    dict.print_check()


def train(chunks_X, chunks_y, tagset, word2vec, save_model):
    # TODO it should be possible to set epochs parameters during fitting, not constructing
    disambiguator = MorphDisambiguator(tagset, word2vec)

    with timing('Training disambiguator'):
        disambiguator.fit(chunks_X, chunks_y, epochs=6)

    with timing('Saving disambiguator model'):
        disambiguator.save_model(save_model)


def load_train(analyzed, gold, tagset, word2vec, save_model):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    with timing('Loading train data'):
        chunks_X, chunks_y = load_chunks_set(analyzed, gold, limit=4096)

    with timing('Loading word vectors'):
        word2vec = KeyedVectors.load_word2vec_format(word2vec, limit=10)

    check(chunks_X, chunks_y, tagset)

    train(chunks_X, chunks_y, tagset, word2vec, save_model)


if __name__ == '__main__':
    load_train(
        analyzed='c:/data/train/nkjp/poleval2017/train-analyzed.xml',
        gold='c:/data/train/nkjp/poleval2017/train-gold.xml',
        tagset=nkjp.tagset,
        word2vec='c:/data/nkjp+wiki-forms-all-300-skipg-ns.txt',
        save_model='data/disambiguation2.h5'
    )
