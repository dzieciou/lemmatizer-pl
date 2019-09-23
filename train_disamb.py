"""
Train MorphDisambiguator and save its model to file.
"""

import logging
import os

from gensim.models import KeyedVectors
from tensorflow.python.util import deprecation
from tqdm import tqdm

from lemmatizer import nkjp
from lemmatizer.disamb import MorphDisambiguator
from lemmatizer.eval import timing
from xces import load_chunks_set

deprecation._PRINT_DEPRECATION_WARNINGS = False

def configure_logs():
    # TODO Into class https://stackoverflow.com/questions/20666764/python-logging-how-to-ensure-logfile-directory-is-created
    os.makedirs('logs', exist_ok=True)
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'file_handler': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': os.path.join('logs', 'training.log'),
                'encoding': 'utf8'
            },
            'console_handler': {
                'class': 'lemmatizer.eval.TqdmLoggingHandler',
                'level': 'DEBUG',
                'formatter': 'standard'
            },
        },
        'loggers': {
            '': {
                'handlers': ['file_handler', 'console_handler'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(logging_config)



def check_ctag(ctag, tagset):
    if not ctag in tagset.valid_ctags:
        raise ValueError(f'Invalid ctag: {ctag}')

@timing
def check_chunks(chunks, tagset):
    for chunk in tqdm(chunks, desc='Checking chunks against tagset'):
        for token in chunk.tokens:
            if not token.ctags is None:
                for ctag in token.ctags:
                    check_ctag(ctag, tagset)



def check(chunks_X, chunks_y, tagset):

        check_chunks(chunks_X, tagset)
        check_chunks(chunks_y, tagset)

    # TODO Move this to checks for PosTagger or MorphAnalyzer
    #      disambiguator is using only tagset, not dictionary
    # with timing('Checking dictionary against tagset'):
    #    dict.print_check()


@timing
def train(chunks_X, chunks_y, tagset, word2vec, save_model):
    # TODO it should be possible to set epochs parameters during fitting, not constructing
    disambiguator = MorphDisambiguator(tagset, word2vec)

    disambiguator.fit(chunks_X, chunks_y, epochs=6)

    disambiguator.save_model(save_model)


@timing
def load_train(analyzed, gold, tagset, word2vec, save_model):

    chunks_X, chunks_y = load_chunks_set(analyzed, gold, limit=4096)

    word2vec = KeyedVectors.load_word2vec_format(word2vec, limit=10000)

    check(chunks_X, chunks_y, tagset)

    train(chunks_X, chunks_y, tagset, word2vec, save_model)


if __name__ == '__main__':
    configure_logs()
    load_train(
        analyzed='c:/data/train/nkjp/poleval2017/train-analyzed.xml',
        gold='c:/data/train/nkjp/poleval2017/train-gold.xml',
        tagset=nkjp.tagset,
        word2vec='c:/data/nkjp+wiki-forms-all-300-skipg-ns.txt',
        save_model='data/disambiguation2.h5'
    )
