from itertools import product


class Tagset:

    def __init__(self, categories, combinations, lexemes = None):
        self.categories = categories
        self.combinations = combinations
        self.valid_ctags = self._generate_values(combinations)
        self.all_pos = tuple(combinations.keys())
        self.lexemes = lexemes
        if not lexemes is None:
            self.lexem_by_pos = self._build_lexem_index(lexemes)

    def _without_noval(self, category):
        return category[:-1]

    def _generate_values(self, combinations):
        valid_ctags = set()
        for pos, combination in combinations.items():
            values_combinations = product(
                *[self._without_noval(category) for category in combination])
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

    def _optional(cat_values):
        '''
        Syntactic sugar to mark certain category optional for a given part of
        speech.
        '''
        return (None,) + cat_values


class Entry:
    """
    Dictionary entry.
    """

    def __init__(self, orth, lemma, ctag, rest):
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

    @classmethod
    def parse(cls, line, expand_tags=True):
        orth, lemma, ctag, *rest = line.strip().split('\t')
        entries = []
        if expand_tags:
            # Expand ctags with . notation into multiple ctags
            for seq in product(*[v.split('.') for v in ctag.split(':')]):
                ctag = ':'.join(seq)
                entries.append(Entry(orth, lemma, ctag, rest))
        else:
            entries.append(Entry(orth, lemma, ctag, rest))

        return entries

    def __str__(self):
        return f'{{orth={self.orth}, lemma={self.lemma}, ctag={self.ctag}, rest={self.rest}}}'

    def __repr__(self):
        return self.__str__()


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
            category = category[:-1]
            tagset_values.update(category)

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
        return self.lookup[token]


def load_dict(fpath, tagset, limit=None, expand_tags=True):
    """
    Load dictionary from text file.
    :param fpath: str
                  The file path to the saved dictionary text file.
    :param limit: int, optional
                  Sets a maximum number of dictionary entries to read from
                  the file. The default, None, means read all.
    :return: Dictionary
    """
    lookup = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        # Skip comments
        comments = 0
        while comments < 3:
            line = f.readline()
            if line.startswith('#'):
                comments += 1

        if limit:
            lines = [next(f) for _ in range(limit)]
        else:
            lines = f

        for line in lines:
            entries = Entry.parse(line, expand_tags)
            orth = entries[0].orth
            if orth not in lookup:
                lookup[orth] = []
            lookup[orth].extend(entries)

    return Dictionary(lookup, tagset)
