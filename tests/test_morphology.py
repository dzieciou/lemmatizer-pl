import pytest

from lemmatizer.morphology import Tagset, optional, Category, Dictionary


def get_sample_tagset():
    number = Category('number', ('sg', 'pl', 'NOVAL'))
    gender = Category('gender', ('m1', 'm2', 'm3', 'f', 'NOVAL2'))
    another_category = Category('another_category', ('x', 'y', 'NOVAL3'))
    yet_another_category = Category('yet_another_category',
                                    ('k', 'l', 'NOVAL4'))

    lexemes = {
        'adj': ('adj',),
        'noun': ('subst',),
    }

    combinations = {
        'adj': [number,
                optional(another_category),
                optional(yet_another_category)],
        'subst': [number, gender]
    }

    pos = Category('pos', combinations.keys())
    categories = (pos, number, gender, another_category, yet_another_category)

    return Tagset(categories, combinations, lexemes)


def test_parse_ctag():
    assert get_sample_tagset().parse_ctag('adj:x') == {
        'pos': 'adj',
        'another_category': 'x'
    }

def test_cast_to_lexeme():
    tagset = get_sample_tagset()
    assert tagset.cast_to_lexeme('subst') == 'noun'

def test_dictionary():

    lookup = {'Camel': ['A'], 'camel': ['B'], 'case': ['C']}
    tagset = get_sample_tagset()
    dict = Dictionary(lookup, tagset)
    assert dict['Camel'] == ['A']
    assert dict['camel'] == ['B']
    assert dict['Case'] == ['C']
    with pytest.raises(KeyError):
        assert dict['Dog']


def test_Tagset():
    assert get_sample_tagset().valid_ctags == set({
        'adj:pl',
        'adj:pl:k',
        'adj:pl:l',
        'adj:pl:x',
        'adj:pl:x:k',
        'adj:pl:x:l',
        'adj:pl:y',
        'adj:pl:y:k',
        'adj:pl:y:l',
        'adj:sg',
        'adj:sg:k',
        'adj:sg:l',
        'adj:sg:x',
        'adj:sg:x:k',
        'adj:sg:x:l',
        'adj:sg:y',
        'adj:sg:y:k',
        'adj:sg:y:l',
        'subst:pl:f',
        'subst:pl:m1',
        'subst:pl:m2',
        'subst:pl:m3',
        'subst:sg:f',
        'subst:sg:m1',
        'subst:sg:m2',
        'subst:sg:m3',
    })
