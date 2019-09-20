from lemmatizer.analyze import MorphAnalyzer

from lemmatizer import Chunk, Token
from lemmatizer.morphology import Dictionary, DictEntry


def test_MorphAnalyzer():
    known_word = 'pomidorów'
    chunks = [
        Chunk([
            Token(known_word)
        ])
    ]

    dict = Dictionary({known_word: [DictEntry(known_word, 'pomidor', 'xyz', '')]})
    analyzer = MorphAnalyzer(dict)

    analyzer.analyze(chunks)
    token = chunks[0].tokens[0]
    assert token.lemmas == ['pomidor']
    assert token.ctags == ['xyz']


def test_MorphAnalyzer_unknown_word():
    unknown_word = 'Kotkowicach'
    chunks = [
        Chunk([
            Token(unknown_word)
        ])
    ]

    empty_dict = Dictionary({})
    analyzer = MorphAnalyzer(empty_dict)

    analyzer.analyze(chunks)
    token = chunks[0].tokens[0]
    assert token.lemmas == [unknown_word]
    assert token.ctags == ['ign']
