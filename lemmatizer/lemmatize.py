import logging

from lemmatizer.analyze import MorphAnalyzer
from lemmatizer.disamb import MorphDisambiguator
from lemmatizer.eval import timing
from lemmatizer.text import Chunk, Token

log = logging.getLogger(__name__)

class Lemmatizer:

    @classmethod
    def create(cls, dict, word2vec, model_fpath=None):
        analyzer = MorphAnalyzer(dict)
        disambiguator = MorphDisambiguator(dict.tagset, word2vec)
        lemmatizer = Lemmatizer(analyzer, disambiguator)
        if not model_fpath is None:
            lemmatizer.load_model(model_fpath)
        return lemmatizer

    def __init__(self, analyzer, disambiguator):
        self._analyzer = analyzer
        self._disambiguator = disambiguator

    def to_chunk(self, tokens):
        tokens = [Token(orth) for orth in tokens]
        return Chunk(tokens)

    # TODO Rename to lemmatize
    @timing
    def tag(self, chunks):
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        elif isinstance(chunks, list):
            if isinstance(chunks[0], str):
                # List of strings
                chunks = [self.to_chunk(chunks)]
            elif isinstance(chunks[0], Chunk):
                pass
            else:
                raise ValueError(
                    'Should provide a chunk or a list of chunks or strings')
        else:
            raise ValueError(
                'Should provide a chunk or a list of chunks or strings')

        self._analyzer.analyze(chunks)
        pred_chunks = self._disambiguator.predict(chunks)
        for chunk, pred_chunk in zip(chunks, pred_chunks):
            for token, pred_token in zip(chunk.tokens, pred_chunk.tokens):
                disamb_ctag = pred_token.disamb_ctag
                candidate_ctags = token.ctags
                if 'ign' in candidate_ctags:
                    log.warning(f'No lemma can be found for {token.orth}. '
                                f'Falling back to original form.')
                    disamb_lemma = token.orth
                else:
                    index = candidate_ctags.index(disamb_ctag)
                    disamb_lemma = token.lemmas[index]
                pred_token.disamb_lemma = disamb_lemma

        return pred_chunks

    def load_model(self, fpath):
        self._disambiguator.load_model(fpath)
