"""
Demonstrates how to use PosTagger with a pretrained model for disambiguation.
"""

from gensim.models import KeyedVectors

from toygger import PosTagger, Chunk, Token
from toygger.polimorf import load_dict


def create_postagger(dict, word2vec, model):
    dict = load_dict(dict)
    word2vec = KeyedVectors.load_word2vec_format(word2vec)
    return PosTagger.create(dict, word2vec, model)


def to_chunk(tokens):
    tokens = [Token(orth) for orth in tokens]
    return Chunk(tokens)


if __name__ == '__main__':
    tagger = create_postagger(
        dict='data/polimorf-20190818.tab',
        word2vec='data/nkjp+wiki-forms-all-300-skipg-ns.txt',
        model='data/disambiguation.h5'
    )

    tokenized = 'Ala ma kota'.split(' ')
    chunks = [to_chunk(tokenized)]
    tagged = tagger.tag(chunks)

    print(tagged)
