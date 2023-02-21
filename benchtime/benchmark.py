import pandas as pd
from datatime import load_dataset
from benchtime.wrappers import Wrapper
import awkward as ak
from numpy.typing import NDArray
from benchtime.evaluation import score_classification
from typing import Optional, Callable, Type
from tqdm.auto import tqdm
import pathlib
import json
from datetime import datetime


def single_benchmark(
    model: Wrapper,
    X_train: ak.Array,
    y_train: NDArray,
    X_test: ak.Array,
    y_test: NDArray,
    scoring_function: Callable,
    model_name: str = "",
    dataset_name: str = "",
    date_id: Optional[bool] = False,
    additional_id: str = "",
    save_folder: Optional[str] = None,
) -> dict:

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = scoring_function(y_test, y_pred)
    scores["fit_time"] = model.fit_time_
    scores["predict_time"] = model.predict_time_
    scores["transform_time"] = model.transform_time_

    if save_folder is not None:
        save_folder = pathlib.Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        if not date_id:
            date_id = ""
        else:
            date_id = datetime.today().strftime("%Y%m%d%H%M%S")
        filename = f"{dataset_name}_{model_name}_{additional_id}_{date_id}_.json"
        with open(save_folder / filename, "w") as write_file:
            json.dump(scores, write_file)
    return scores


def benchmark_model(
    wrapper: Type[Wrapper],
    model,
    dataset_names: list[str],
    scoring_function: Callable,
    save_folder: Optional[str] = None,
    model_params: Optional[dict] = None,
    model_name: str = "",
    date_id: Optional[bool] = False,
    additional_id: str = "",
    conversion_function: Callable = lambda x: x,
):
    scores_df = pd.DataFrame()
    pbar = tqdm(dataset_names)
    for dataset_name in pbar:
        pbar.set_description(dataset_name)
        dataset = load_dataset(dataset_name)
        X_train, y_train, X_test, y_test = dataset()
        wrapped_model = wrapper(
            model=model,
            model_params=model_params,
            conversion_function=conversion_function,
        )
        scores = single_benchmark(
            model=wrapped_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            scoring_function=scoring_function,
            model_name=model_name,
            dataset_name=dataset_name,
            date_id=date_id,
            additional_id=additional_id,
            save_folder=save_folder,
        )
        scores_df = pd.concat([scores_df, pd.DataFrame(scores, index=[dataset_name])])
    return scores_df
