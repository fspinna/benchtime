import awkward as ak
from numpy.typing import NDArray
from typing import Callable, Optional
from datetime import datetime


class Wrapper:
    def __init__(
        self,
        model,
        model_params: Optional[dict] = None,
        conversion_function: Callable = lambda x: x,
    ):
        if model_params is None:
            model_params = dict()
        self.model = model(**model_params)
        self.conversion_function = conversion_function

        self.fit_time_ = None
        self.predict_time_ = None
        self.transform_time_ = None

    def fit(self, X: ak.Array, y: NDArray):
        X_conv = self.conversion_function(X)
        start_time = datetime.now()
        self.model.fit(X_conv, y)
        self.fit_time_ = (datetime.now() - start_time).total_seconds()
        return self

    def predict(self, X: ak.Array) -> NDArray:
        X_conv = self.conversion_function(X)
        start_time = datetime.now()
        y = self.model.predict(X_conv)
        self.predict_time_ = (datetime.now() - start_time).total_seconds()
        return y

    def predict_proba(self, X: ak.Array) -> NDArray:
        return self.model.predict_proba(self.conversion_function(X))

    def transform(self, X: ak.Array) -> NDArray:
        X_conv = self.conversion_function(X)
        start_time = datetime.now()
        X_transformed = self.model.transform(X_conv)
        self.transform_time_ = (datetime.now() - start_time).total_seconds()
        return X_transformed
