import types

from sklearn.base import TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper


class KerasInputFormatter(_BaseComposition, TransformerMixin):
    # Solution to passing multiple inputs from Scikit to Keras
    # Copies from https://github.com/keras-team/keras/issues/9001#issuecomment-365522923

    def __init__(self, transformers):
        """
        :param transformers: a dictionary of named transformers
        """
        self.transformers = transformers

    def fit(self, X, y=None):
        for n, t in self.transformers:
            t.fit(X, y)

        return self

    def transform(self, X):
        return {n: t.transform(X) for n, t in self.transformers}

    def get_params(self, deep=True):
        return self._get_params('transformers', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('transformers', **kwargs)
        return self


class KerasClassifierMultipleOutputs(BaseWrapper):

    def _create_model(self):
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

    def load_model_weights(self, fpath):
        self._create_model()
        self.model.load_weights(fpath)

    def fit(self, x, y, **fit_params):
        """
        Yes, successive calls to fit will incrementally train the model.
        :param x:
        :param y:
        :param fit_params:
        :return:
        """
        if not hasattr(self, 'model') or self.model is None:
            self._create_model()
        callbacks = [
            # FIXME This results in a bug
            #       tensorflow.python.eager.core._FallbackException: This
            #       function does not handle the case of the path where all
            #       inputs are not already EagerTensors.
            #TensorBoard(
            #    log_dir='log_dir',
            #    histogram_freq=1
            #)
        ]
        history = self.model.fit(x, y, callbacks=callbacks, **fit_params)
        return history

    def predict(self, x, **kwargs):
        #kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        y_prob = self.model.predict(x, **kwargs)
        classes = [y.argmax(axis=-1) for y in y_prob]
        # TODO Return dictionary of outputs instead, using the approach below
        #      https://github.com/keras-team/keras/issues/2422#issuecomment-383662647
        return classes
        #return self.classes_[classes]

    def transform(self, X):
        self._create_model()
