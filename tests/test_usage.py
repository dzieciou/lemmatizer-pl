import logging
import re

from gensim.models import KeyedVectors

from eval import timing
from lemmatizer import Lemmatizer
from lemmatizer.polimorf import load_dict
from lemmatizer.text import Chunk, Token


def tokenize(text):
    def to_chunks(s): return re.split('\.|\?|!', s)

    def to_tokens(s): return s.split()

    chunks = [Chunk([Token(t)
                     for t
                     in to_tokens(chunk)])
              for chunk
              in to_chunks(text)]

    return chunks


log = logging.getLogger()
log.setLevel(logging.DEBUG)

def test_usage():

    with timing('Loading dictionary entries'):
        dict = load_dict('data/dict/polimorf-20190818.tab',
                         limit=5000)

    with timing('Loading word vectors'):
        word_vectors = KeyedVectors.load_word2vec_format(
            'data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=5000)

    with timing('Initializing POS tagger'):
        posTagger = Lemmatizer.create(dict, word_vectors)
        posTagger.load_model('data/disambiguation.h5')

    text = '5 kilogramów pomidorów trafiło do kuchnii. Zostały ugotowane na miękko.'
    chunks = tokenize(text)
    assert chunks[0].tokens[1].orth == 'kilogramów'

    print(chunks)

    posTagger.tag(chunks)
    assert chunks[0].tokens[1].disamb_lemma == 'kilogram'
    assert chunks[0].tokens[1].disamb_tag.startswith('noun')

    print(chunks)
