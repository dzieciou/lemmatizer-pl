import logging
import math
import random
import string
from operator import itemgetter

import Levenshtein
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from toygger.encoders import (
    WordEmbedEncoder,
    CTagsEncoder,
    DisambCTagEncoder
)
from toygger.indexing import build_categories_index
from toygger.keras import KerasInputFormatter, KerasClassifierMultipleOutputs
from toygger.models import create_model
from toygger.morphology import Dictionary
from toygger.text import Chunk, Token

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class MorphDisambiguator(BaseEstimator, TransformerMixin):

    def __init__(self, tagset, word2vec, bucket_size=2048, epochs=1,
        correct_preds=True):
        self.tagset = tagset
        self.word2vec = word2vec
        self.bucket_size = bucket_size
        self.epochs = epochs
        self._pipeline = None
        self._y_tranformer = None
        self.correct_preds = correct_preds

    def _check_setup(self):
        if self._pipeline is None:
            self._setup()

    def _setup(self):
        categories = self.tagset.categories
        self._pipeline = self._create_pipeline(categories, self.word2vec)
        self._y_transformer = DisambCTagEncoder(categories)
        self._y_corrector = CTagCorrector(categories, self.tagset)

    def _create_pipeline(self, categories, word2vec):
        categories_index = build_categories_index(categories)
        biLSTM = KerasClassifierMultipleOutputs(
            build_fn=create_model,
            categories=categories,
            categories_index=categories_index,
            word2vec=word2vec)
        pipeline = Pipeline([
            ('inputs', KerasInputFormatter([
                ('word_vec', WordEmbedEncoder(word2vec)),
                ('ctags_vec', CTagsEncoder(categories))])),
            ('clf', biLSTM)
        ])
        return pipeline

    def _make_buckets(self, chunks_X, chunks_y):

        chunks_X = sorted(chunks_X, key=lambda chunk: len(chunk.tokens))
        chunks_y = sorted(chunks_y, key=lambda chunk: len(chunk.tokens))
        buckets = []
        buckets_count = math.ceil(len(chunks_X) / self.bucket_size)
        for bucket_id in list(range(buckets_count)):
            start = bucket_id * self.bucket_size
            end = start + self.bucket_size
            bucket = (chunks_X[start:end], chunks_y[start:end])
            buckets.append(bucket)
        return buckets

    def score(self, y, y_pred):
        return self.word_accuracy_score(y, y_pred)

    def set_params(self, **params):
        attrs = list(params.keys())
        for attr in attrs:
            setattr(self, attr, params.pop(attr))

        self._pipeline.set_params(**params)

    def fit(self, chunks_X, chunks_y, **fit_params):
        if len(chunks_X) != len(chunks_y):
            raise ValueError()

        self._check_setup()

        self.set_params(**fit_params)

        buckets = self._make_buckets(chunks_X, chunks_y)
        for epoch in range(self.epochs):
            log.debug(f'Epoch {epoch + 1}/{self.epochs}...')
            random.shuffle(buckets)
            for bucket_id, (chunks_X, chunks_y) in enumerate(buckets):
                log.debug(f'Bucket {bucket_id + 1}/{len(buckets)}...')
                y = self._y_transformer.fit_transform(chunks_y)
                self._pipeline.fit(chunks_X, y,
                                   clf__epochs=1,
                                   clf__batch_size=128)

    def transform(self, chunks_X):
        '''
        Disambiguate ctags for given chunks of tokens.
        :param chunks_X:
        :return: list of list of predicted ctags for given chunks of tokens.
        '''
        y_pred = self._pipeline.predict(chunks_X)
        pred_chunks = self._y_transformer.inverse_transform(y_pred)
        if self.correct_preds:
            pred_chunks = self._y_corrector.correct(chunks_X, pred_chunks)

        for chunk_X, pred_chunk in zip(chunks_X, pred_chunks):
            for token_X, pred_token in zip(chunk_X.tokens, pred_chunk.tokens):
                pred_token.orth = token_X.orth
                pred_token.lemmas = token_X.lemmas
                pred_token.ctags = token_X.ctags

        return pred_chunks

    predict = transform

    def save_model(self, fpath):
        self._get_model().model.save_weights(fpath)

    def load_model(self, fpath):
        self._check_setup()
        self._get_model().load_model_weights(fpath)

    def _get_model(self):
        return self._pipeline.named_steps['clf']


class CTagCorrector:

    def __init__(self, categories, tagset):
        self.categories = categories
        self.tagset = tagset
        self.value2code, self.code2value = self._build_indexes()

    def correct(self, chunks, y_pred_chunks):
        '''
        Returns corrected ctags.
        :param chunks: Analyzed chunks of tokens
        :param predicted_ctags: list of list of ctags predicted for tokens
        :return: list of list of corrected ctags
        '''
        out_chunks = []
        for chunk, y_pred_chunk in zip(chunks, y_pred_chunks):
            out_chunk = Chunk()
            out_chunks.append(out_chunk)
            for token, y_pred_token in zip(chunk.tokens, y_pred_chunk.tokens):
                predicted_ctag = y_pred_token.disamb_ctag
                correct_ctag = self.correct_token(token, predicted_ctag)
                out_token = Token(token.orth, disamb_ctag=correct_ctag)
                out_chunk.tokens.append(out_token)
        return out_chunks

    def correct_token(self, token, predicted_ctag):
        candidate_ctags = token.ctags
        if 'ign' in candidate_ctags:
            candidate_ctags = self.tagset.valid_ctags

        if predicted_ctag in candidate_ctags:
            return predicted_ctag
        else:
            return self._most_similar(predicted_ctag, candidate_ctags)

    def _most_similar(self, predicted_ctag, candidate_ctags):
        pred_ctag_code = self._encode(predicted_ctag)
        cand_ctags_codes = list(map(self._encode, candidate_ctags))
        distances = [
            (ctag_code, Levenshtein.distance(ctag_code, pred_ctag_code))
            for ctag_code
            in cand_ctags_codes
        ]
        _, (ctag_code, distance) = min(enumerate(distances), key=itemgetter(1))
        return self._decode(ctag_code)

    def _encode(self, ctag):
        values = ctag.split(':')
        codes = [self.value2code[value] for value in values]
        code = ''.join(codes)
        return code

    def _decode(self, code):
        codes = [code[i:i + 2] for i in range(0, len(code), 2)]
        values = [self.code2value[code] for code in codes]
        ctag = ':'.join(values)
        return ctag

    def _build_indexes(self):
        value2code = {}
        code2value = {}
        alphabet = (c for c in string.printable)
        for category in self.categories:
            category_code = next(alphabet)
            for value in category.values:
                if not value.startswith('NO'):
                    value_code = category_code + next(alphabet)
                    value2code[value] = value_code
                    code2value[value_code] = value

        return value2code, code2value


class MorphAnalyzer:
    '''
    Performs morphological analysis on text chunks. Precisely, for each token
    in each chunk provides a list of possible lemmas and their morphological
    categories using a provided dictionary.
    '''

    def __init__(self, dict: Dictionary):
        '''
        Creates morphological analyzer.
        :param dict: morphological dictionary
        '''
        self.dict = dict

    def analyze(self, chunks):
        '''
        Updates a list of given chunks with morphological information
        :param chunks: text chunks to update
        '''
        for chunk in chunks:
            for token in chunk.tokens:
                try:
                    entries = self.dict[token.orth]
                    token.ctags = [e.ctag for e in entries]
                    token.lemmas = [e.lemma for e in entries]
                except KeyError:
                    token.ctags = ['ign']
                    token.lemmas = [token.orth]


class PosTagger:

    @classmethod
    def create(cls, dict, word2vec, model_fpath=None):
        analyzer = MorphAnalyzer(dict)
        disambiguator = MorphDisambiguator(dict.tagset, word2vec)
        tagger = PosTagger(analyzer, disambiguator)
        if not model_fpath is None:
            tagger.load_model(model_fpath)
        return tagger

    def __init__(self, analyzer, disambiguator):
        self._analyzer = analyzer
        self._disambiguator = disambiguator

    def to_chunk(self, tokens):
        tokens = [Token(orth) for orth in tokens]
        return Chunk(tokens)

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
                index = token.ctags.index(disamb_ctag)
                disamb_lemma = token.lemmas[index]
                pred_token.disamb_lemma = disamb_lemma

        return pred_chunks

    def load_model(self, fpath):
        self._disambiguator.load_model(fpath)
