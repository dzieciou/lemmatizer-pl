"""
Poliform tagset has been defined based on the following resources:

- NKJP tagset: http://nkjp.pl/poliqarp/help/ense2.html

- Narodowy Korpus Języka Polskiego, praca zbiorowa pod redakcją Adama
  Przepiórkowskiego, Mirosława Bańko, Rafała L. Górskiego, Barbary
  Lewandowskiej-Tomaszczyk, Wydawnictwo Naukowe PWN, Warszawa 2012
  http://nkjp.pl/settings/papers/NKJP_ksiazka.pdf

"""

from lemmatizer import morphology as morph

from lemmatizer.morphology import Tagset, Category, optional

lexemes = {
    'verb': (
        'fin', 'bedzie', 'aglt', 'praet', 'impt', 'imps', 'inf', 'pcon', 'pant',
        'ger', 'pact', 'ppas'),
    'winien': ('winien',),
    'pred': ('pred',),
    'noun': ('subst', 'depr'),
    'adj': ('adj', 'adja', 'adjp', 'adjc'),
    'adv': ('adv',),
    'num': ('num', 'numcol'),
    'ppron12': ('ppron12',),
    'ppron3': ('ppron3',),
    'siebie': ('siebie',),
    'prep': ('prep',),
    'conj': ('conj',),
    'comp': ('comp',),
    'interj': ('interj',),
    'brev': ('brev',),
    'burk': ('burk',),
    'qub': ('qub',),
    'xxx': ('xxx',),
}

number = Category('number', ('sg', 'pl', 'NONUM'))
case = Category('case', ('nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc',
                         'NOCASE'))
gender = Category('gender', ('m1', 'm2', 'm3', 'f', 'n', 'NOGEND'))
accommodability = Category('accommodability', ('rec', 'congr', 'NOREC'))
person = Category('person', ('pri', 'sec', 'ter', 'NOPERS'))
accentability = Category('accentability', ('akc', 'nakc', 'NOACC'))
post_prepositionality = Category('post_prepositionality', ('praep', 'npraep',
                                                           'NOPRP'))
degree = Category('degree', ('pos', 'com', 'sup', 'NODEG'))
aspect = Category('aspect', ('perf', 'imperf', 'NOASP'))
agglutination = Category('agglutination', ('agl', 'nagl', 'NOAGL'))
vocalicity = Category('vocalicity', ('wok', 'nwok', 'NOVOC'))
negation = Category('negation', ('aff', 'neg', 'NOAFF'))
fullstoppedness = Category('fullstoppedness', ('pun', 'npun', 'NOPUN'))

combinations = {
    'adj': (number, case, gender, degree),
    'adja': (),
    'adjc': (),
    'adjp': (),
    'adv': (optional(degree),),
    'aglt': (number, person, aspect, vocalicity),
    'bedzie': (number, person, aspect),
    'brev': (fullstoppedness,),
    'burk': (),
    'comp': (),
    'conj': (),
    'depr': (number, case, gender),
    'fin': (number, person, aspect),
    'ger': (number, case, gender, aspect, negation),
    'ign': (),
    'imps': (aspect,),
    'impt': (number, person, aspect),
    'inf': (aspect,),
    'interj': (),
    'interp': (),
    'num': (number, case, gender, accommodability),
    'numcol': (number, case, gender, accommodability),
    'pact': (number, case, gender, aspect, negation),
    'pant': (aspect,),
    'pcon': (aspect,),
    'ppas': (number, case, gender, aspect, negation),
    'ppron12': (number, case, gender, person, optional(accentability)),
    'ppron3': (number, case, gender, person, optional(accentability),
               optional(post_prepositionality)),
    'praet': (number, gender, aspect, optional(agglutination)),
    'pred': (),
    'prep': (case, optional(vocalicity)),
    'qub': (optional(vocalicity),),
    'siebie': (case,),
    'subst': (number, case, gender),
    'winien': (number, gender, aspect),
    'xxx': (),
}

pos = Category('pos', tuple(combinations.keys()))

categories = (
    pos, number, case, gender, accommodability, person, accentability,
    post_prepositionality, degree, aspect, agglutination, vocalicity, negation,
    fullstoppedness
)

tagset = Tagset(categories, combinations, lexemes)


def load_dict(fpath, limit=None):
    return morph.load_morfeusz2_dict(fpath, tagset, limit)
