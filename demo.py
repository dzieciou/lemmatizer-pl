"""
Demonstrates how to use PosTagger with a pretrained model for disambiguation.
"""

from gensim.models import KeyedVectors

from lemmatizer.lemmatize import Lemmatizer
from lemmatizer.nkjp import load_dict


def create_postagger(dict, word2vec, model):
    dict = load_dict(dict)
    word2vec = KeyedVectors.load_word2vec_format(word2vec, limit=100)
    return Lemmatizer.create(dict, word2vec, model)


if __name__ == '__main__':
    tagger = create_postagger(
        dict='data/dict/polimorf2nkjp-20190818.tab',
        word2vec='data/nkjp+wiki-forms-all-300-skipg-ns.txt',
        model='data/disambiguation-best.h5'
    )

    # FIXME Tagger does not handle dots and other punctation marks
    tokenized = 'Ala ma kota'.split(' ')
    tagged = tagger.tag(tokenized)
    for chunk in tagged:
        for token in chunk.tokens:
            print(f'{token.orth}\t{token.disamb_lemma}\t{token.disamb_ctag}')
