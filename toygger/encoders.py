from abc import abstractmethod, ABC
from collections import defaultdict

import Levenshtein
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin

from toygger.text import Chunk, Token
from toygger.indexing import build_categories_index


class ChunkEncoder(BaseEstimator, TransformerMixin, ABC):
    '''
    Transforms list of chunks into a numpy array of shape
    (len(chunks), maxlen, token_vector_size)
    '''

    def transform(self, chunks):
        out_chunks = []
        for chunk in chunks:
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
        for word in vocab:
            for suffix_length in range(1, len(word)):
                suffix = word[-suffix_length:]
                words_by_suffix[suffix].add(word)
        return words_by_suffix

    def _inverse_transform_token(self, token_y):
        # FIXME Looks non pythonic here
        raise NotImplementedError


class CTagsEncoder(ChunkEncoder):

    def __init__(self, categories):
        self.categories = categories;
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

    def transform(self, chunks):
        '''
        Transforms to all outputs
        :param chunks:
        :return:
        '''
        outputs = defaultdict(list)
        for chunk in chunks:
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

    def inverse_transform(self, outputs):

        # FIXME it would be better if outputs was a dictionary, but
        #       it looks it's not
        chunks = []
        for chunk_output in zip(*outputs):
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
