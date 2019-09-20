import json


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=None)


class Token:
    def __init__(self, orth, lemmas=None, ctags=None, disamb_lemma=None,
        disamb_ctag=None):
        self.orth = orth
        self.lemmas = lemmas
        self.ctags = ctags
        self.disamb_lemma = disamb_lemma
        self.disamb_ctag = disamb_ctag

    def default(self, o):
        return o.__dict__

    def __str__(self):
        return to_json(self.__dict__)

    def __repr__(self):
        return self.__str__()


class Chunk:
    def __init__(self, tokens=None):
        if tokens is None:
            tokens = []
        self.tokens = tokens

    def __str__(self):
        return to_json({'tokens': [t.__dict__ for t in self.tokens]})

    def __repr__(self):
        return self.__str__()