from toygger import polimorf, nkjp


# TODO Mentiond this very rough conversion to get things done
# TODO Mention about MACA
# TODO Document convertion rules
# TODO Add tag collapsing
# FIXME Note we do not add original preamble again

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


def convert_dict(in_fpath, out_fpath, convert):
    dict = polimorf.load_dict(in_fpath, limit=None)
    with open(out_fpath, 'wb') as f:
        for orth, entries in dict.lookup.items():
            for e in entries:
                try:
                    e.ctag = convert(e.ctag)
                except IncompatibleCTag:
                    # skip writing whole entry
                    continue
                except ValueError as ex:
                    raise ValueError(f'Invalid ctag for {e.orth}', ex)
                else:
                    line = f'{e.orth}\t{e.lemma}\t{e.ctag}\n'
                    f.write(line.encode('utf-8'))


convert_dict('data/dict/polimorf-20190818.tab',
             'data/dict/polimorf2nkjp-20190818.tab',
             polimorf2nkjp)
