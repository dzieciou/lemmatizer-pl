"""
Poliform tagset has been defined based on the work:

    Marcin Wolinski, Automatyczna analiza składnikowa języka polskiego,
    Wydawnictwa Uniwersytetu Warszawskiego 2019
	https://doi.org/10.31338/uw.9788323536147

and updated by comparing with actual tags in Poliform dictionary.

    http://download.sgjp.pl/morfeusz/20190818/polimorf-20190818.tab.gz

"""

from lemmatizer import morphology as morph
from lemmatizer.morphology import Tagset, Category, optional

lexemes = {
    'verb': (
        'fin', 'bedzie', 'aglt', 'praet', 'cond', 'impt', 'imps',
        'inf', 'pcon', 'pant', 'ger', 'pact', 'ppas'),
    'winien': ('winien',),
    'pred': ('pred',),
    'noun': ('subst', 'depr'),
    'adj': ('adj', 'adja', 'adjp', 'adjc'),
    'adv': ('adv',),
    'num': ('num', 'numcomp'),
    'ppron12': ('ppron12',),
    'ppron3': ('ppron3',),
    'siebie': ('siebie',),
    'prep': ('prep',),
    'conj': ('conj',),
    'comp': ('comp',),
    'interj': ('interj',),
    'part': ('part',),
    'frag': ('frag',),
}

number = Category('number', ('sg', 'pl', 'NO_NUMBER'))
case = Category('case', (
    'nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc', 'NO_CASE'))
gender = Category('gender',
                  ('m1', 'm2', 'm3', 'f', 'n', 'NO_GENDER'))
subgender = Category('subgender',
                     ('col', 'ncol', 'pt', 'NO_SUBGENDER'))
person = Category('person', ('pri', 'sec', 'ter', 'NO_PERSON'))
degree = Category('degree', ('pos', 'com', 'sup', 'NO_DEGREE'))
aspect = Category('aspect', ('imperf', 'perf', 'NO_ASPECT'))
negation = Category('negation', ('aff', 'neg', 'NO_NEGATION'))
accentability = Category('accentability',
                         ('akc', 'nakc', 'NO_ACCENTABILITY'))
post_prepositionality = Category('post_prepositionality', (
    'npraep', 'praep', 'NO_POST_PREPOSITIONALITY'))
accommodability = Category('accommodability',
                           ('congr', 'rec', 'NO_ACCOMODABILITY'))
agglutination = Category('agglutination',
                         ('agl', 'nagl', 'NO_AGGLUTINATION'))
vocalicity = Category('vocalicity',
                      ('nwok', 'wok', 'NO_VOCALICITY'))
fullstoppedness = Category('fullstoppedness',
                           ('pun', 'npun', 'NO_FULLSTOPPEDNESS'))

combinations = {
    'brev': (fullstoppedness,),
    'fin': (number, person, aspect),
    'bedzie': (number, person, aspect),
    'aglt': (number, person, aspect, vocalicity),
    'praet': (number, gender, optional(person), aspect,
              optional(agglutination), optional(aspect)),
    'cond': (number, gender, person, aspect),
    'impt': (number, person, aspect),
    'imps': (aspect,),
    'inf': (aspect,),
    'pcon': (aspect,),
    'pant': (aspect,),
    'ger': (number, case, gender, aspect, negation),
    'pact': (number, case, gender, aspect, negation),
    'pacta': (),
    'ppas': (number, case, gender, aspect, negation),
    'winien': (number, gender, optional(person), aspect),
    'pred': (),
    'subst': (number, case, gender, optional(subgender)),
    'depr': (number, case, gender),
    'adj': (number, case, gender, degree),
    'adja': (),
    'adjp': (case,),
    'adjc': (),
    'adv': (optional(degree),),
    'num': (number, case, gender, accommodability, optional(subgender)),
    'numcomp': (),
    'ppron12': (number, case, gender, person, optional(accentability)),
    'ppron3': (number, case, gender, person, accentability,
               post_prepositionality),
    'siebie': (optional(case),),
    'prep': (case, optional(vocalicity)),
    'conj': (),
    'comp': (),
    'interj': (),
    'part': (),
    'frag': (),
}

pos = Category('pos', tuple(combinations.keys()))

categories = (
    pos, number, case, gender, subgender, person, degree, aspect, negation,
    accentability, post_prepositionality, accommodability, agglutination,
    vocalicity, fullstoppedness
)

tagset = Tagset(categories, combinations, lexemes)


def load_dict(fpath, limit=None):
    return morph.load_morfeusz2_dict(fpath, tagset, limit)
