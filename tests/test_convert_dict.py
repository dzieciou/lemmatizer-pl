from convert_dict import collapse
from lemmatizer import nkjp
from lemmatizer.morphology import DictEntry


def parse(line):
    orth, lemma, ctag = line.split(' ')
    return DictEntry(orth, lemma, ctag)


def test_ctag_collapse():
    input = [
        'abchaskim X adj:sg:dat:n',
        'abchaskim abchaski adj:pl:dat:m1',
        'abchaskim abchaski adj:pl:dat:m2',
        'abchaskim abchaski adj:pl:dat:m3',
        'abchaskim abchaski adj:pl:dat:f',
        'abchaskim abchaski adj:pl:dat:n',
        'abchaskim abchaski adj:sg:inst:m1',
        'abchaskim abchaski adj:sg:inst:m2',
        'abchaskim abchaski adj:sg:inst:m3'
    ]

    expected = [
        'abchaskim X adj:sg:dat:n',
        'abchaskim abchaski adj:pl:dat:f.m1.m2.m3.n',
        'abchaskim abchaski adj:sg:inst:m1.m2.m3'
    ]

    input = list(map(parse, input))
    expected = list(map(parse, expected))

    collapsed = collapse(nkjp.tagset, 'abchaskim', input)
    assert collapsed == expected


def test_ctag_collapse_with_no_gender():
    input = [
        'abchasko abchaski adja',
        'abchasko abchasko adv:pos',
        'abchasku abchaski adjp'
    ]

    expected = [
        'abchasko abchaski adja',
        'abchasko abchasko adv:pos',
        'abchasku abchaski adjp'
    ]

    input = list(map(parse, input))
    expected = list(map(parse, expected))

    collapsed = collapse(nkjp.tagset, 'abchasko', input)
    assert collapsed == expected


def test_ctag_collapse_multiple_categories():
    input = [
        'Aachen Aachen subst:pl:nom:m3',
        'Aachen Aachen subst:pl:gen:m3',
        'Aachen Aachen subst:pl:dat:m3',
        'Aachen Aachen subst:sg:nom:m3',
        'Aachen Aachen subst:sg:gen:m3',
        'Aachen Aachen subst:sg:dat:m3'
    ]

    expected = [
        'Aachen Aachen subst:pl.sg:dat.gen.nom:m3'
    ]

    input = list(map(parse, input))
    expected = list(map(parse, expected))

    collapsed = collapse(nkjp.tagset, 'Aachen', input)
    assert collapsed == expected
