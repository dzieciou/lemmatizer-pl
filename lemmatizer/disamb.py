import logging
import math
import random
import string
from abc import abstractmethod, ABC
from collections import defaultdict
from operator import itemgetter

import Levenshtein
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.layers import (
    Input,
    concatenate,
    Masking,
    Bidirectional,
    LSTM,
    Dense
)
from tensorflow.python.keras.models import (
    Model
)
from tensorflow.python.keras.utils import plot_model
from tqdm import tqdm

from lemmatizer.eval import timing
from lemmatizer.indexing import build_categories_index
from lemmatizer.keras import KerasInputFormatter, KerasClassifierMultipleOutputs
from lemmatizer.text import Chunk, Token

log = logging.getLogger(__name__)

HIDDEN_LAYERS = 2
HIDDEN_LAYER_DIMENSION = 384


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

    @timing
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

    @timing
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


class ChunkEncoder(BaseEstimator, TransformerMixin, ABC):
    '''
    Transforms list of chunks into a numpy array of shape
    (len(chunks), maxlen, token_vector_size)
    '''

    @timing
    def transform(self, chunks):
        out_chunks = []
        encoder_name = type(self).__name__
        for chunk in tqdm(chunks, desc=f'Encoding chunks with {encoder_name}'):
            out_chunk = []
            for token in chunk.tokens:
                vector = self._transform_token(token)
                out_chunk.append(vector)
            out_chunks.append(out_chunk)

        out_chunks = pad_sequences(out_chunks,
                                   padding='post',
                                   dtype='float',
                                   value=self._default_value())

        return out_chunks

    def inverse_transform(self, y):

        out_rows = []
        for chunk_y in y:
            out_row = []
            for token_y in chunk_y:
                out_row.append(self._inverse_transform_token(token_y))
            out_rows.append(out_row)

        return out_rows

    @abstractmethod
    def _default_value(self):
        return

    @abstractmethod
    def _transform_token(self, token):
        return

    @abstractmethod
    def _inverse_transform_token(self, token_y):
        return


class WordEmbedEncoder(ChunkEncoder):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.words_by_suffix = self._group_words_by_suffix(self.word2vec.vocab)
        self.vector_size = self.word2vec.wv.vector_size

    def fit(self, chunks, *args):
        return self

    def _transform_token(self, token):
        return self._word2vector(token.orth)

    def _default_value(self):
        return np.zeros(self.vector_size)

    def _word2vector(self, word):
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            for suffix_length in reversed(range(1, len(word))):
                suffix = word[-suffix_length:]
                if suffix in self.words_by_suffix:
                    alternatives = self.words_by_suffix[suffix]
                    alternatives = sorted(
                        (Levenshtein.distance(word, alternative), alternative)
                        for alternative in alternatives
                    )
                    min_dist = alternatives[0][0]
                    alternative_vectors = [
                        self.word2vec[alternative]
                        for dist, alternative in alternatives
                        if dist == min_dist
                    ]
                    return np.mean(alternative_vectors, axis=0)

        return np.random.uniform(-1, 1, self.vector_size)

    def _group_words_by_suffix(self, vocab):
        words_by_suffix = defaultdict(set)
        for word in tqdm(vocab, desc='Building word2vec suffix index'):
            for suffix_length in range(1, len(word)):
                suffix = word[-suffix_length:]
                words_by_suffix[suffix].add(word)
        return words_by_suffix

    def _inverse_transform_token(self, token_y):
        # FIXME Looks non pythonic here
        raise NotImplementedError


class CTagsEncoder(ChunkEncoder):

    def __init__(self, categories):
        self.categories = categories
        self.categories_index = build_categories_index(categories)

    def fit(self, chunks, *args):
        return self

    def _transform_ctags(self, ctags):
        vector = np.zeros(len(self.categories_index))
        values = set()
        for ctag in ctags:
            values.update(ctag.split(':'))

        for category in self.categories:
            active_values = values.intersection(category.values)
            if active_values:
                self._multi_hot_encoding(vector,
                                         active_values,
                                         self.categories_index)
            else:
                novalue = category.values[-1]
                vector[self.categories_index[novalue]] = 1

        return vector

    def _default_value(self):
        return np.zeros(len(self.categories_index))

    def _transform_token(self, token):
        return self._transform_ctags(token.ctags)

    def _multi_hot_encoding(self, tensor, values, index):
        for v in values:
            tensor[index[v]] = 1

    def _inverse_transform_token(self, token_y):
        raise NotImplementedError


class MultiOutputsChunkEncoder(BaseEstimator, TransformerMixin, ABC):
    '''
    Transform list of chunks to dictionary of numpy arrays, where each entry
    contains sumples for one one output. A numpy_array for a given output_id has shape:
    (len(chunks), max_tokens, size(output_name))
    '''

    def fit(self, chunks, *args):
        return self

    def __init__(self, outputs_names):
        self.outputs_names = outputs_names

    @timing
    def transform(self, chunks):
        '''
        Transforms to all outputs
        :param chunks:
        :return:
        '''
        outputs = defaultdict(list)
        encoder_name = type(self).__name__
        for chunk in tqdm(chunks, desc=f'Encoding chunks with {encoder_name}'):
            for output_id in self.outputs_names:
                outputs[output_id].append([])
            for token in chunk.tokens:
                ys = self._transform_token(token)
                for output_id in self.outputs_names:
                    output = outputs[output_id]
                    y = ys[output_id]
                    output[-1].append(y)

        for output_id in self.outputs_names:
            output = outputs[output_id]
            output = pad_sequences(output,
                                   padding='post',
                                   dtype='float',
                                   value=self._default_value(output_id))
            outputs[output_id] = output

        return outputs

    @timing
    def inverse_transform(self, outputs):

        # FIXME it would be better if outputs was a dictionary, but
        #       it looks it's not
        encoder_name = type(self).__name__
        chunks = []
        total = len(outputs[0])
        for chunk_output in tqdm(zip(*outputs),
                                 desc=f'Decoding chunks with {encoder_name}',
                                 total=total):
            chunk = Chunk()
            chunks.append(chunk)
            for token_output in zip(*chunk_output):
                ctag = self._inverse_transform_token(token_output)
                token = Token(orth='', disamb_ctag=ctag)
                chunk.tokens.append(token)

        return chunks

    @abstractmethod
    def _transform_token(self, token, output_id):
        pass

    @abstractmethod
    def _output_size(self, output_id):
        pass

    @abstractmethod
    def _inverse_transform_token(self, token_y):
        pass


class DisambCTagEncoder(MultiOutputsChunkEncoder):

    def __init__(self, categories):
        outputs_names = [category.name for category in categories]
        super().__init__(outputs_names)
        self.categories = categories
        self.encoder = CTagsEncoder(categories)

    def _transform_token(self, token):
        tag_vector = self.encoder._transform_ctags([token.disamb_ctag])
        outputs = self.split_by_output(tag_vector)
        return outputs

    def _output_size(self, output_id):
        return len(self.categories[output_id])

    def _default_value(self, output_id):
        category_id = self.outputs_names.index(output_id)
        category = self.categories[category_id]
        return np.zeros(len(category.values))

    def split_by_output(self, tag_vector):
        offset = 0
        outputs = {}
        for outputs_ids, category in zip(self.outputs_names, self.categories):
            l = len(category.values)
            outputs[outputs_ids] = tag_vector[offset:(offset + l)]
            offset += l
        return outputs

    def _inverse_transform_token(self, ys):
        values = []
        for category in self.categories:
            output_id = self.outputs_names.index(category.name)
            y = ys[output_id]
            value = category.values[y]
            if not value.startswith('NO'):
                values.append(value)
        return ':'.join(values)


def create_model(categories, categories_index, word2vec):
    # First dimension of inputs is variable (None) because length of
    # sentences can vary.

    ctags_vec = Input(
        shape=(None, len(categories_index)),
        dtype='float32',
        name='ctags_vec'
    )

    word_vec = Input(
        shape=(None, word2vec.vector_size),
        dtype='float32',
        name='word_vec'
    )

    all_inputs = concatenate([ctags_vec, word_vec])

    masked_inputs = Masking(mask_value=0., )(all_inputs)

    biLSTM = Bidirectional(LSTM(
        HIDDEN_LAYER_DIMENSION,
        return_sequences=True, dropout=0.5)
    )(masked_inputs)

    for _ in range(HIDDEN_LAYERS - 1):
        biLSTM = Bidirectional(LSTM(
            HIDDEN_LAYER_DIMENSION,
            return_sequences=True,
            dropout=0.5)
        )(biLSTM)

    predicted_tag_categories = [
        Dense(
            len(category.values),
            activation='softmax',
            name=f'{category.name}')(biLSTM)
        for category
        in categories
    ]

    model = Model(
        inputs=[ctags_vec, word_vec],
        outputs=predicted_tag_categories
    )

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # TODO Should we remove it? Or enable with some debug flag?
    plot_model(model, to_file='model.png')

    return model
