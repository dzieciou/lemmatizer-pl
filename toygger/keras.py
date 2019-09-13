import types

from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper
from sklearn.base import TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition


class KerasInputFormatter(_BaseComposition, TransformerMixin):
    # Solution to passing multiple inputs from Scikit to Keras
    # Copies from https://github.com/keras-team/keras/issues/9001#issuecomment-365522923

    def __init__(self, transformers):
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
        # FIXME what if it is already created
        self._create_model()
        history = self.model.fit(x, y, **fit_params)
        return history

    def predict(self, x, **kwargs):
        #kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        y_prob = self.model.predict(x, **kwargs)
        classes = [y.argmax(axis=-1) for y in y_prob]
        return classes
        #return self.classes_[classes]

    def transform(self, X):
        self._create_model()
