import os
import shutil
import typing as t
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache

import typer
import yaml
import joblib
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.model_selection import GridSearchCV

import data
import model
import metrics

app = typer.Typer()


@lru_cache(None)
def _read_csv(filepath):
    return pd.read_csv(filepath)


class CsvDatasetReader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def __call__(self):
        return _read_csv(self.filepath)


@app.command()
def train(config_file: str):
    hyperparams = _load_config(config_file, "hyperparams")
    split = "train"
    X, y = _get_dataset(_load_config(config_file, "data"), splits=[split])[split]
    estimator = model.build_estimator(hyperparams)
    estimator.fit(X, y)
    output_dir = _load_config(config_file, "export")["output_dir"]
    version = _save_versioned_estimator(estimator, hyperparams, output_dir)
    return version


def _get_dataset(data_config, splits):
    filepath = data_config["filepath"]
    reader = CsvDatasetReader(filepath)
    return data.get_dataset(reader=reader, splits=splits)


def _save_versioned_estimator(
    estimator: BaseEstimator, hyperparams: t.Dict[str, t.Any], output_dir: str
):
    version = str(datetime.now(timezone.utc).replace(second=0, microsecond=0))
    model_dir = os.path.join(output_dir, version)
    os.makedirs(model_dir, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(model_dir, "model.joblib"))
        _save_yaml(hyperparams, os.path.join(model_dir, "params.yml"))
    except Exception as e:
        typer.echo(f"Coudln't serialize model due to error {e}")
        shutil.rmtree(model_dir)
    return version


@app.command()
def find_hyperparams(
    config_file: str,
    train_best_model: bool = typer.Argument(False),
):
    search_config = _load_config(config_file, "search")
    param_grid = search_config["grid"]
    n_jobs = search_config["jobs"]
    metric = _load_config(config_file, "metrics")[0]
    dummy_hyperparams = {name: {} for name in param_grid.keys()}
    estimator = model.build_estimator(dummy_hyperparams)
    scoring = metrics.get_scoring_function(metric["name"], **metric["params"])
    gs = GridSearchCV(
        estimator,
        _param_grid_to_sklearn_format(param_grid),
        n_jobs=n_jobs,
        scoring=scoring,
        verbose=3,
    )
    split = "train"
    X, y = _get_dataset(_load_config(config_file, "data"), splits=[split])[split]
    gs.fit(X, y)
    hyperparams = _param_grid_to_custom_format(gs.best_params_)
    estimator = model.build_estimator(hyperparams)
    output_dir = _load_config(config_file, "export")["output_dir"]
    _save_versioned_estimator(estimator, hyperparams, output_dir)


def _param_grid_to_sklearn_format(param_grid):
    return {
        f"{name}__{pname}": pvalues
        for name, params in param_grid.items()
        for pname, pvalues in params.items()
    }


def _param_grid_to_custom_format(param_grid):
    grid = {}
    for name, values in param_grid.items():
        estimator_name, param_name = name.split("__", maxsplit=1)
        if estimator_name not in grid:
            grid[estimator_name] = {}
        grid[estimator_name][param_name] = values
    return grid


@app.command()
def eval(
    config_file: str,
    model_version: str,
    splits: t.List[str] = ["test"],
):
    output_dir = _load_config(config_file, "export")["output_dir"]
    saved_model = os.path.join(output_dir, model_version, "model.joblib")
    estimator = joblib.load(saved_model)
    dataset = _get_dataset(_load_config(config_file, "data"), splits=splits)
    report = defaultdict(list)
    all_metrics = _load_config(config_file, "metrics")
    for name, (X, y) in dataset.items():
        y_pred = estimator.predict(X)
        for m in all_metrics:
            metric_name, params = m["name"], m["params"]
            fn = metrics.get_metric_function(metric_name, **params)
            value = float(fn(y, y_pred))
            report[metric_name].append({"split": name, "value": value})
    reports_dir = _load_config(config_file, "reports")["dir"]
    _save_yaml(
        dict(report),
        os.path.join(reports_dir, f"{model_version}.yml"),
    )


def _load_config(filepath: str, key: str):
    content = _load_yaml(filepath)
    config = content[key]
    return config


@lru_cache(None)
def _load_yaml(filepath: str) -> t.Dict[str, t.Any]:
    with open(filepath, "r") as f:
        content = yaml.load(f)
    return content


def _save_yaml(content: t.Dict[str, t.Any], filepath: str):
    with open(filepath, "w") as f:
        yaml.dump(content, f)


if __name__ == "__main__":
    app()
