from collections import namedtuple
from itertools import product
from typing import List

from smart_open import open
from tqdm import tqdm

Category = namedtuple('Category', ['name', 'values'])


def optional(category):
    '''
    Syntactic sugar to mark certain category optional for a given part of
    speech.
    '''
    values = (None,) + category.values
    return Category(name=category.name, values=values)


class Tagset:

    def __init__(self, categories, combinations, lexemes=None):
        self.categories = categories
        self.value2category = self._build_value2category_index(categories)
        self.combinations = combinations
        self.valid_ctags = self._generate_values(combinations)
        self.all_pos = tuple(combinations.keys())
        self.lexemes = lexemes
        if not lexemes is None:
            self.lexem_by_pos = self._build_lexem_index(lexemes)

    def _build_value2category_index(self, categories: List[Category]):
        index = {}
        for category in categories:
            for value in category.values:
                if not value.startswith('NO'):
                    index[value] = category.name
        return index

    def _without_noval(self, category):
        return category[:-1]

    def _generate_values(self, combinations):
        valid_ctags = set()
        for pos, combination in combinations.items():
            values_combinations = product(
                *[self._without_noval(category.values) for category in
                  combination])
            for seq in values_combinations:
                ctag = ':'.join(filter(None, (pos,) + seq))
                valid_ctags.add(ctag)
        return valid_ctags

    def _build_lexem_index(self, lexemes):
        index = {}
        for lexeme, poses in lexemes.items():
            for pos in poses:
                index[pos] = lexeme
        return index

    def check_ctag(self, ctag):
        if ctag not in self.valid_ctags:
            raise ValueError(f'Invalid ctag: {ctag}')

    def get_category(self, value):
        if '.' in value:
            value = value.split('.')[0]
        return self.value2category[value]

    def parse_ctag(self, ctag):
        values = ctag.split(':')
        return {self.get_category(value): value for value in values}

    def build_ctag(self, parsed):
        ctag = ':'.join([value for key, value in parsed.items()])
        self.check_ctag(ctag)
        return ctag

    def cast_to_lexeme(self, pos_flexeme):
        return self.lexem_by_pos[pos_flexeme]



class DictEntry:
    """
    Dictionary entry.
    """

    def __init__(self, orth, lemma, ctag, rest=None):
        """
        :param orth:
        :param lemma:
        :param ctag:
        :param rest:
        """
        self.orth = orth
        self.lemma = lemma
        self.ctag = ctag
        self.rest = rest

    def __str__(self):
        return f'{{orth={self.orth}, lemma={self.lemma}, ctag={self.ctag}, rest={self.rest}}}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, DictEntry):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.orth == other.orth \
               and self.lemma == other.lemma \
               and self.ctag == other.ctag \
               and self.rest == other.rest


class Dictionary:
    """
    Dictionary of words and their morphological categories
    """

    def __init__(self, lookup, tagset=None):
        self.tagset = tagset
        self.lookup = lookup
        self.tagset = tagset

    def get_invalid_entries(self):
        for orth, entries in self.lookup.items():
            for e in entries:
                if not e.ctag in self.tagset.valid_ctags:
                    yield orth, e.ctag

    def compare_tagset_with_dict(self):
        dict_values = set()
        for _, entries in self.lookup.items():
            for e in entries:
                v = e.ctag.split(':')
                dict_values.update(v)

        tagset_values = set()
        tagset_values.update(self.tagset.all_pos)
        for category in self.tagset.categories:
            # without no_val
            values = category.values[:-1]
            tagset_values.update(values)

        values_missing = sorted(dict_values.difference(tagset_values))
        values_unused = sorted(tagset_values.difference(dict_values))
        return values_missing, values_unused

    def print_check(self):
        print("Invalid dictionary entries:")
        for orth, ctag in self.get_invalid_entries():
            print(f'Invalid entry for {orth} with ctag {ctag}')

        values_missing, values_unused = self.compare_tagset_with_dict()
        print("Values missing in tagset:", values_missing)
        print("Values unused from tagset:", values_unused)

    def __getitem__(self, token):
        '''
        Returns a list of dictionary entries for a given token.

        KeyError is thrown if no entry can be found
        :param token:
        :return:
        '''

        if token in self.lookup:
            return self.lookup[token]

        return self.lookup[token.lower()]

def load_dict(fpath, tagset, limit=None, expand_tags=True, sep='\t',
    handle_preamble=None):
    """
    Load dictionary from text file.
    :param fpath: str
                  The file path to the saved dictionary text file.
    :param tagset: Tagset
                  Tagset of the dictionary.
    :param limit: int, optional
                  Sets a maximum number of dictionary entries to read from
                  the file. The default, None, means read all.
    :param expand_tags: bool
                  Sets whether to expand tags with . dot notation to multiple
                  entries.
    :param sep: str
                  Separator of fields in a single line.
    :return: Dictionary
    """

    def parse(line):
        orth, lemma, ctag, *rest = line.strip().split(sep)
        entries = []
        if expand_tags:
            # Expand ctags with . notation into multiple ctags
            for seq in product(*[v.split('.') for v in ctag.split(':')]):
                ctag = ':'.join(seq)
                entries.append(DictEntry(orth, lemma, ctag, rest))
        else:
            entries.append(DictEntry(orth, lemma, ctag, rest))
        return entries

    lookup = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        if not handle_preamble is None:
            handle_preamble(f)

        if limit:
            lines = [next(f) for _ in range(limit)]
        else:
            lines = f

        for line in tqdm(lines, unit=' line', desc=f'Reading dictionary {fpath}'):
            entries = parse(line)
            orth = entries[0].orth
            if orth not in lookup:
                lookup[orth] = []
            lookup[orth].extend(entries)

    return Dictionary(lookup, tagset)


def load_morfeusz1_dict(fpath, tagset, limit=None):
    def skip_first_line(f):
        f.readline()

    return load_dict(fpath, tagset, limit, expand_tags=True, sep=' ',
                     handle_preamble=skip_first_line)


def load_morfeusz2_dict(fpath, tagset, limit=None):
    def skip_comments(f):
        comments = 0
        while comments < 3:
            line = f.readline()
            if line.startswith('#'):
                comments += 1

    return load_dict(fpath, tagset, limit, expand_tags=True, sep='\t',
                     handle_preamble=skip_comments)
