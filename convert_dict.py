from collections import defaultdict
from itertools import groupby

from toygger import polimorf, nkjp
# TODO Mentiond this very rough conversion to get things done
# TODO Mention about MACA
# TODO Document convertion rules
# TODO Add tag collapsing
# FIXME Note we do not add original preamble again
from toygger.morphology import DictEntry


class IncompatibleCTag(ValueError):
    pass


def polimorf2nkjp(ctag):
    d = polimorf.tagset.parse_ctag(ctag)

    # Some words do not have corresponding annotations in NKJP
    if d['pos'] in ['numcomp']:
        # Dla liczebników wprowadzono także fleksem reprezentujący formę występującą w złożeniach numcomp. W większości wypadków nie pojawia się ona w wynikach analizy, bo słowa takie jak czterodrzwiowemu są traktowane jako
        # formy leksemu czterodrzwiowy. Forma numcomp może pojawić się samodzielnie przy analizie pierwszego członu konstrukcji takich jak cztero- albo pięciodrzwiowy.
        # https://www.wuw.pl/data/include/cms/Automatyczna_analiza_skladnikowa_Wolinski_Marcin_2019.pdf
        raise IncompatibleCTag()

    # Some POS are named differently in NKJP
    d['pos'] = {
        'frag': 'burk',
        'part': 'qub',
        'cond': 'praet'
    }.get(d['pos'], d['pos'])

    # Some categories are not present at all in NKJP
    d.pop('subgender', None)

    # Some categories are not present for selected POS in NKJP
    if d['pos'] == 'adjp':
        d.pop('case', None)
    if d['pos'] == 'praet':
        d.pop('person', None)
    if d['pos'] == 'winien':
        d.pop('person', None)

    ctag = nkjp.tagset.build_ctag(d)
    return ctag


def collapse_ctags(tagset, orth, lemma, entries):
    for category in ['number', 'gender', 'case' ]:
        entries = collapse_ctags_by_category(tagset, orth, lemma, entries, category)
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
    for orth, entries in lookup.items():
        merged.extend(collapse(tagset, orth, entries))
    return merged

def convert_entries(lookup, convert):
    converted = defaultdict(list)
    for orth, entries in lookup.items():
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
    with open(fpath, 'wb') as f:
        for e in entries:
            line = f'{e.orth}\t{e.lemma}\t{e.ctag}\n'
            f.write(line.encode('utf-8'))

def convert_dict(in_fpath, out_fpath, convert):
    dict = polimorf.load_dict(in_fpath)
    converted = convert_entries(dict.lookup, convert)
    merged = collapse_all(nkjp.tagset, converted)
    write_dict(merged, out_fpath)




if __name__ == '__main__':
    convert_dict('data/dict/polimorf-20190818.tab',
                 'data/dict/polimorf2nkjp-20190818.tab',
                 polimorf2nkjp)
