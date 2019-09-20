import numpy as np
from gensim.models import KeyedVectors

from lemmatizer import Token, Chunk
from lemmatizer.disamb import WordEmbedEncoder, CTagsEncoder, DisambCTagEncoder
from lemmatizer.keras import KerasInputFormatter
from lemmatizer.morphology import Category
from lemmatizer.polimorf import tagset


def test_WordEmbedEncoder():
    # TODO we should mock this
    word2vec = KeyedVectors.load_word2vec_format(
        'data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=50)

    chunks = [
        Chunk([
            Token('5'),
            Token('kilogramów'),
            Token('pomiodorów')
        ])
    ]

    encoder = WordEmbedEncoder(word2vec)
    X = encoder.fit_transform(chunks)
    assert type(X) == np.ndarray
    assert X.shape == (1,3,300)


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
    assert type(X) == np.ndarray
    assert X.shape == (1, 3, 88)


def test_KerasInputFormatter():
    # TODO Move somewhere else
    # TODO we should mock this
    word2vec = KeyedVectors.load_word2vec_format(
        'data/nkjp+wiki-forms-all-300-skipg-ns.txt', limit=5)

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
    assert isinstance(y, dict)
    pos = y['pos']
    assert type(pos) == np.ndarray
    assert pos.shape == (2, 5, 35)


def test_DisambCTagEncoder_decoding():
    categories = (
        Category('1', ('a', 'b', 'c', 'NOVAL')),
        Category('2', ('d', 'e'))
    )

    y_pred = [
        [
            np.array([1, 0,3])
        ],
        [
            np.array([0, 1,0],),
        ]
    ]

    encoder = DisambCTagEncoder(categories)

    predicted_chunks = encoder.inverse_transform(y_pred)
    predicted_ctags = [token.disamb_ctag
                       for chunk
                       in predicted_chunks
                       for token
                       in chunk.tokens]
    assert predicted_ctags ==['b:d', 'a:e', 'd']


def test_CTagCodeEncoder():
    pass
