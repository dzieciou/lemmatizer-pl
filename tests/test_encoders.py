import numpy as np
from gensim.models import KeyedVectors

from toygger import Token, Chunk
from toygger.encoders import WordEmbedEncoder, CTagsEncoder, DisambCTagEncoder
from toygger.keras import KerasInputFormatter
from toygger.polimorf import tagset


def test_WordEmbedEncoder():
    # TODO we should mock this
    word2vec = KeyedVectors.load_word2vec_format(
        '../data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=5000)

    chunks = [
        Chunk([
            Token('5'),
            Token('kilogramów'),
            Token('pomiodorów')
        ])
    ]

    encoder = WordEmbedEncoder(word2vec)
    X = encoder.fit_transform(chunks)
    assert type(X) == list
    assert type(X[0]) == list
    assert type(X[0][0]) == np.ndarray


def test_CTagsEncoder():
    chunks = [
        Chunk([
            Token('5', ctags=['brev:pun', 'conj', 'prep:nom']),
            Token('kilogramów', ctags=['qub']),
            Token('pomidorów', ctags=['brev:pun', 'conj', 'prep:nom'])
        ])
    ]

    encoder = CTagsEncoder(tagset.categories)

    X = encoder.fit_transform(chunks)
    assert type(X) == list
    assert type(X[0]) == list
    assert type(X[0][0]) == np.ndarray
    assert np.max(X[0][0]) == 1.0  # at least one value set


def test_KerasInputFormatter():
    # TODO Move somewhere else
    # TODO we should mock this
    word2vec = KeyedVectors.load_word2vec_format(
        '../data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=5000)

    u = KerasInputFormatter([
        ('word2vec', WordEmbedEncoder(word2vec)),
        ('tag2vec', CTagsEncoder(tagset.categories))])

    chunks = [
        Chunk([
            Token('5', ctags=['brev:pun', 'conj', 'prep:nom']),
            Token('kilogramów', ctags=['qub']),
            Token('pomidorów', ctags=['brev:pun', 'conj', 'prep:nom'])
        ])
    ]

    X = u.fit_transform(chunks)

    print(X)


def test_DisambCTagEncoder():
    chunks = [
        Chunk([
            Token('5', disamb_ctag='brev:pun'),
            Token('kilogramów', disamb_ctag='qub'),
            Token('pomidorów', disamb_ctag='conj')
        ]),
        Chunk([
            Token('5', disamb_ctag='brev:pun'),
            Token('kilogramów', disamb_ctag='qub'),
            Token('pomidorów', disamb_ctag='conj'),
            Token('i', disamb_ctag='conj'),
            Token('ogórków', disamb_ctag='conj')
        ])
    ]

    encoder = DisambCTagEncoder(tagset.categories)

    y = encoder.fit_transform(chunks)
    assert type(y) == list
    assert len(y) == len(tagset.categories)
    assert type(y[0]) == np.ndarray
    assert type(y[0][0]) == np.ndarray
    assert type(y[0][0][0]) == np.ndarray
    assert np.count_nonzero(y[0][0][0]) == 1

    print(y)


def test_DisambCTagEncoder_decoding():
    categories = (
        ('a', 'b', 'c'),
        ('d', 'e')
    )

    y_pred = [
        [
            [np.array([0.3, 0.6, 0.1]), np.array([0.2, 0.8])],
            [np.array([0.6, 0.3, 0.1]), np.array([0.8, 0.2])]
        ],
        [
            [np.array([0.6, 0.3, 0.1]), np.array([0.8, 0.2])],
            [np.array([0.3, 0.6, 0.1]), np.array([0.2, 0.8])],
        ]
    ]

    encoder = DisambCTagEncoder(categories)

    predictions = encoder.inverse_transform(y_pred)

    assert predictions == [['b:e', 'a:d'], ['a:d', 'b:e']]


def test_CTagCodeEncoder():
    pass
