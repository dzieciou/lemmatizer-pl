"""
Converts dictionary from one tagset to another automatically. This is a very
rough conversion and possibly lossy without additional human annotator input.

An alternative might be using/consulting more advanced tools written by
linguists, e.g. MACA (Morphological Analysis Converter and Aggregator)
http://nlp.pwr.wroc.pl/redmine/projects/libpltagger/wiki.

Polimorf to NKJP tagset conversion rules have been defined by consulting the
following resources:

- Marcin Wolinski, Automatyczna analiza składnikowa języka polskiego,
  Wydawnictwa Uniwersytetu Warszawskiego 2019
  https://doi.org/10.31338/uw.9788323536147
- Morfeusz online demo: http://morfeusz.sgjp.pl/demo/
- Morfeusz source code: http://download.sgjp.pl/morfeusz/
- Narodowy Korpus Języka Polskiego, praca zbiorowa pod redakcją Adama
  Przepiórkowskiego, Mirosława Bańko, Rafała L. Górskiego, Barbary
  Lewandowskiej-Tomaszczyk, Wydawnictwo Naukowe PWN, Warszawa 2012
  http://nkjp.pl/settings/papers/NKJP_ksiazka.pdf
- NKJP tagset: http://nkjp.pl/poliqarp/help/ense2.html
- Poliqarp search engine for NKJP data: http://nkjp.pl/poliqarp/
- MACA conversion rules, http://nlp.pwr.wroc.pl/redmine/projects/libpltagger/wiki
"""

from collections import defaultdict
from itertools import groupby

from tqdm import tqdm

from lemmatizer import polimorf, nkjp
from lemmatizer.morphology import DictEntry


class IncompatibleCTag(ValueError):
    pass


def polimorf2nkjp(ctag):
    d = polimorf.tagset.parse_ctag(ctag)

    # Some words do not have corresponding annotations in NKJP tagset.
    if d['pos'] in ['numcomp']:
        # numcomp correponds to "cztero" in phrases like "cztero- albo
        # pięciodrzwiowy". Words like "czterodrzwiowy" are not split by Morfeusz
        # and hence are not treated as numcomp. There is no correponding
        # lexem in NKJP tagset. The closest would be num, however, it requires
        # additional categories like number, case, gender, etc.
        raise IncompatibleCTag()

    # Some POS are named differently in NKJP tagset.
    d['pos'] = {
        # frag stands for part of a phraseme and describes words that are not
        # present seperately in other contexts, e.g. "wznak" exists only in
        # "na wznak". In NKJP tagset they are called burkinostki (burk).
        'frag': 'burk',
        # Following MACA conversion rule...
        'part': 'qub',
        # cond describes conditional forms like "widziałbym". In NKJP tagset
        # they are split into three tokens, e.g.
        #   widział [widzieć:praet:sg:m1:imperf]
        #   by      [by:qub]
        #   m       [być:aglt:sg:pri:imperf:nwok]
        # Our morphological analyzer does not split words into smaller token
        # (i.e. so-called agglutination is not considered).
        'cond': 'praet',
        # In Polimorf it is used to describe composed forms like "ssąco" in
        # "ssąco-tłocząca".
        'pacta': 'adja'
    }.get(d['pos'], d['pos'])

    # Some categories are not present at all in NKJP tagset.
    # Subgender (pl. przyrodzaj) in some tagsets is described with custom gender
    # values (n1, p1, p2). In NKJP tagset this information is not present in
    # any form.
    d.pop('subgender', None)

    # Some categories are not present for selected POS in NKJP tagset.
    if d['pos'] == 'adjp':
        # In NKJP tagset post-prepositional adjectives do not have information
        # about case
        d.pop('case', None)
    if d['pos'] == 'praet':
        # In Polimorf past forms of verbs (pl. przeslik) like "widziałem" are
        # treated as a whole and have information about person, e.g.: "pri" in
        # widziałem	widzieć:praet:sg:m1.m2.m3:pri:imperf
        # In NKJP agglutination is considered, and information about person
        # is included in agglutinate, e.g.:
        #   widział [widzieć:praet:sg:m1:imperf]
        #   by      [by:qub]
        #   m       [być:aglt:sg:pri:imperf:nwok]
        d.pop('person', None)
    if d['pos'] == 'winien':
        d.pop('person', None)

    ctag = nkjp.tagset.build_ctag(d)
    return ctag


def collapse_ctags(tagset, orth, lemma, entries):
    for category in ['number', 'gender', 'case']:
        entries = collapse_ctags_by_category(tagset, orth, lemma, entries,
                                             category)
    return entries


def collapse_ctags_by_category(tagset, orth, lemma, entries, category):
    def common_categories(entry):
        ctag = tagset.parse_ctag(entry.ctag)
        ctag.pop(category, None)
        return ':'.join([value for key, value in ctag.items()])

    merged = []
    to_merge = []
    for entry in entries:
        ctag = tagset.parse_ctag(entry.ctag)
        if category in ctag:
            to_merge.append(entry)
        else:
            merged.append(entry)

    to_merge = sorted(to_merge, key=common_categories)
    for key, group in groupby(to_merge, common_categories):
        values = []
        for entry in group:
            ctag = tagset.parse_ctag(entry.ctag)
            value = ctag[category]
            values.append(value)
        key = tagset.parse_ctag(key)
        key[category] = '.'.join(values)
        key = [key[category.name]
               for category
               in tagset.categories
               if category.name in key]
        ctag = ':'.join(key)
        merged.append(DictEntry(orth, lemma, ctag))

    return merged


def collapse(tagset, orth, entries):
    merged = []
    for lemma, group in groupby(entries, lambda e: e.lemma):
        merged.extend(collapse_ctags(tagset, orth, lemma, group))
    return merged


def collapse_all(tagset, lookup):
    merged = []
    for orth, entries in tqdm(lookup.items(), desc='Merging entries'):
        merged.extend(collapse(tagset, orth, entries))
    return merged


def convert_entries(lookup, convert):
    converted = defaultdict(list)
    for orth, entries in tqdm(lookup.items(), desc='Converting entries'):
        for e in entries:
            try:
                e.ctag = convert(e.ctag)
            except IncompatibleCTag:
                # skip writing whole entry
                continue
            except ValueError as ex:
                raise ValueError(f'Invalid ctag for {e.orth}', ex)
            else:
                converted[orth].append(e)

    return converted


def write_dict(entries, fpath):
    # FIXME Note we do not add original preamble again
    with open(fpath, 'wb') as f:
        for e in tqdm(entries, desc='Writing entries'):
            line = f'{e.orth}\t{e.lemma}\t{e.ctag}\n'
            f.write(line.encode('utf-8'))


def convert_dict(in_fpath, out_fpath, convert):
    dict = polimorf.load_dict(in_fpath)
    converted = convert_entries(dict.lookup, convert)
    merged = collapse_all(nkjp.tagset, converted)
    write_dict(merged, out_fpath)


if __name__ == '__main__':
    convert_dict('data/dicts/polimorf-20190818.tab',
                 'data/dicts/polimorf2nkjp-20190818.tab',
                 polimorf2nkjp)
