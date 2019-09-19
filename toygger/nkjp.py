from toygger import morphology as morph

from toygger.morphology import Tagset, Category, optional

POS_V = Category('pos', ('adj', 'adja', 'adjc', 'adjp', 'adv', 'aglt', 'bedzie',
                         'brev', 'burk', 'comp', 'conj', 'depr', 'fin', 'ger',
                         'ign', 'imps', 'impt', 'inf', 'interj', 'interp',
                         'num', 'numcol', 'pact', 'pacta', 'pant', 'pcon',
                         'ppas', 'ppron12', 'ppron3', 'praet', 'pred', 'prep',
                         'qub', 'siebie', 'subst', 'winien', 'xxx'))
NUM_V = Category('num', ('sg', 'pl', 'NONUM'))
CASE_V = Category('case', ('nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc',
                           'NOCASE'))
GEND_V = Category('gend', ('m1', 'm2', 'm3', 'f', 'n', 'NOGEND'))
REC_V = Category('rec', ('rec', 'congr', 'NOREC'))
PERS_V = Category('pers', ('pri', 'sec', 'ter', 'NOPERS'))
ACC_V = Category('acc', ('akc', 'nakc', 'NOACC'))
PRP_V = Category('prp', ('praep', 'npraep', 'NOPRP'))
DEG_V = Category('deg', ('pos', 'com', 'sup', 'NODEG'))
ASP_V = Category('asp', ('perf', 'imperf', 'NOASP'))
AGL_V = Category('agl', ('agl', 'nagl', 'NOAGL'))
VOC_V = Category('voc', ('wok', 'nwok', 'NOVOC'))
AFF_V = Category('aff', ('aff', 'neg', 'NOAFF'))
PUN_V = Category('pun', ('pun', 'npun', 'NOPUN'))

categories = (
    POS_V,
    NUM_V,
    CASE_V,
    GEND_V,
    REC_V,
    PERS_V,
    ACC_V,
    PRP_V,
    DEG_V,
    ASP_V,
    AGL_V,
    VOC_V,
    AFF_V,
    PUN_V,
)

combinations = {
    'adj': (NUM_V, CASE_V, GEND_V, DEG_V),
    'adja': (),
    'adjc': (),
    'adjp': (),
    'adv': (optional(DEG_V),),
    'aglt': (NUM_V, PERS_V, ASP_V, VOC_V),
    'bedzie': (NUM_V, PERS_V, ASP_V),
    'brev': (PUN_V,),
    'burk': (),
    'comp': (),
    'conj': (),
    'depr': (NUM_V, CASE_V, GEND_V),
    'fin': (NUM_V, PERS_V, ASP_V),
    'ger': (NUM_V, CASE_V, GEND_V, ASP_V, AFF_V),
    'ign': (),
    'imps': (ASP_V,),
    'impt': (NUM_V, PERS_V, ASP_V),
    'inf': (ASP_V,),
    'interj': (),
    'interp': (),
    'num': (NUM_V, CASE_V, GEND_V, REC_V),
    'numcol': (NUM_V, CASE_V, GEND_V, REC_V),
    'pact': (NUM_V, CASE_V, GEND_V, ASP_V, AFF_V),
    'pacta': (),
    'pant': (ASP_V,),
    'pcon': (ASP_V,),
    'ppas': (NUM_V, CASE_V, GEND_V, ASP_V, AFF_V),
    'ppron12': (NUM_V, CASE_V, GEND_V, PERS_V, optional(ACC_V)),
    'ppron3': (NUM_V, CASE_V, GEND_V, PERS_V, optional(ACC_V), optional(PRP_V)),
    'praet': (NUM_V, GEND_V, ASP_V, optional(AGL_V)),
    'pred': (),
    'prep': (CASE_V, optional(VOC_V)),
    'qub': (optional(VOC_V),),
    'siebie': (CASE_V,),
    'subst': (NUM_V, CASE_V, GEND_V),
    'winien': (NUM_V, GEND_V, ASP_V),
    'xxx': (),
}

tagset = Tagset(categories, combinations)


def load_dict(fpath, limit=None):
    return morph.load_morfeusz2_dict(fpath, tagset, limit)
