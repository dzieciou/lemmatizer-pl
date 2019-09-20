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

# TODO Add lexems to cast POS to more generic categories, like in Polimorf tagset

pos = Category('pos', ('adj', 'adja', 'adjc', 'adjp', 'adv', 'aglt', 'bedzie',
                       'brev', 'burk', 'comp', 'conj', 'depr', 'fin', 'ger',
                       'ign', 'imps', 'impt', 'inf', 'interj', 'interp',
                       'num', 'numcol', 'pact', 'pacta', 'pant', 'pcon',
                       'ppas', 'ppron12', 'ppron3', 'praet', 'pred', 'prep',
                       'qub', 'siebie', 'subst', 'winien', 'xxx'))
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

categories = (
    pos,
    number,
    case,
    gender,
    accommodability,
    person,
    accentability,
    post_prepositionality,
    degree,
    aspect,
    agglutination,
    vocalicity,
    negation,
    fullstoppedness,
)

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
    'pacta': (),
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
    'siebie': (case,),
    'subst': (number, case, gender),
    'winien': (number, gender, aspect),
    'xxx': (),
}

tagset = Tagset(categories, combinations)


def load_dict(fpath, limit=None):
    return morph.load_morfeusz2_dict(fpath, tagset, limit)
