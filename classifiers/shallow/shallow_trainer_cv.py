"""
Train a model (GridSearchCV on train+validation) and predict on test.
Multiple combinations FX functions are used, from the `shallow_fx` module.
"""

import argparse
import multiprocessing as mp
from itertools import chain, combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import shallow_fx
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import GridSearchCV

ALL_FX_FUNCS = shallow_fx.collect_all_fxs()


def make_fv_dict(filepath: str) -> dict[str, Any]:
    """
    The FV dict contains the mapping from FX name to FV from all FX functions.
    """
    with open(filepath, "rt") as fp:
        source = fp.read()

    return {fx: getattr(shallow_fx, fx)(source) for fx in ALL_FX_FUNCS}


def make_fv_subset(fv_dict: dict[str, Any], keys: list[str]) -> list:
    """
    Construct a 1-dim FV by concatenating all sub-FVs in the FV dict given by keys.
    """
    out = []

    for k in keys:
        fv = fv_dict[k]
        [out.append, out.extend][isinstance(fv, list)](fv)

    return out


def make_train_test_df(
    input_path: Path,
    sources_dir: Path | None = None,
    fx_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply full FX on the sources and split b/w train & test.
    Only `parseable` samples are considered, i.e. which can be parsed into an AST.

    If the `sources_dir` is missing, then the dataframe is expected to contain a `path` column,
    with the absolute path to the source on-disk.
    """
    df = pd.read_csv(input_path).query("parseable == True")
    assert set(df["label"]) == {"clean", "dirty"}

    if "path" in df.columns:
        logger.info("Found path column in dataframe")
        paths = df["path"].apply(Path)
    else:
        assert sources_dir is not None
        logger.info(f"No path column in dataframe, reading sources from {sources_dir}")
        paths = df["sha256"].apply(lambda h: sources_dir / h)
    assert all(p.exists() for p in paths)

    logger.info(f"Doing FX with {fx_workers} workers on {len(paths)} samples...")
    with mp.Pool(fx_workers) as pool:
        df["fv_dict"] = pool.map(make_fv_dict, paths)

    train_df = df.query('subset == "train"')
    test_df = df.query('subset != "train"')

    return train_df, test_df


def make_Xy(df: pd.DataFrame, fx_funcs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Make X, y vectors from subsetting into FV dict with fx_funcs.
    Example:
        FV dict: {'fx_foo': [...], 'fx_bar': [...], 'fx_baz': [...]}
        fx_keys: ['fx_foo', 'fx_bar']
        out: [fx_foo_out, fx_bar_out]
    """
    X = np.vstack(df["fv_dict"].apply(lambda d: make_fv_subset(d, fx_funcs)))
    y = df["label"].apply(lambda x: {"clean": 0, "dirty": 1}[x]).to_numpy()

    assert X.ndim == 2
    assert len(X) == len(y)

    return X, y


def main(args):
    args.out_models_dir.mkdir(exist_ok=False)

    train_df, test_df = make_train_test_df(args.input_csv, args.sources_dir, args.fx_workers)
    logger.info(f"{len(train_df)=}")
    logger.info(f"{len(test_df)=}")

    base = xgb.XGBClassifier(objective="binary:logistic")

    param_grid = {
        "max_depth": [2, 4, 6, 8, 12],
        "n_estimators": [4, 8, 16, 32, 64, 128],
        "eta": [0.01, 0.05, 0.1, 0.2],
    }

    fx_funcs_list = list(chain.from_iterable(combinations(ALL_FX_FUNCS, r + 1) for r in range(len(ALL_FX_FUNCS))))
    assert len(fx_funcs_list) == 2 ** len(ALL_FX_FUNCS) - 1
    logger.info(f"Using {len(fx_funcs_list)} FX combinations")

    clf = GridSearchCV(
        base,
        param_grid,
        cv=args.cv_folds,
        scoring=args.metric,
        n_jobs=args.cv_workers,
        refit=True,  # refit the model using the best found parameters on the whole dataset
        verbose=1,
    )

    out = []

    for fx_funcs in fx_funcs_list:
        # generate a unique run ID which will map the model to the corresponding results entry in the parquet file
        run_id = uuid4().hex[:8]
        logger.info(f"[{run_id=}] Training with {fx_funcs=}")

        cur = {"run_id": run_id, "fx_funcs": fx_funcs}

        X_train, y_train = make_Xy(train_df, fx_funcs)
        clf.fit(X_train, y_train)

        cur["best_params"] = clf.best_params_
        cur["train_labels"] = y_train
        cur["train_preds"] = clf.predict_proba(X_train)[:, 1]

        X_test, y_test = make_Xy(test_df, fx_funcs)
        cur["test_labels"] = y_test
        cur["test_preds"] = clf.predict_proba(X_test)[:, 1]

        # register current run results
        out.append(cur)
        # save model to file
        joblib.dump(clf, args.out_models_dir / f"model_{run_id}.sklearn")

    logger.info(f"Dumping results to {args.out_parquet_file}")
    pd.DataFrame(out).to_parquet(args.out_parquet_file, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV with dataset info: sha256, label, subset etc.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        required=False,
        help="Dir with the actual sources to be read and parsed. If it's missing, the input CSV must have a `path` column.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds to split train + validation data into.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric to be used for GridSearchCV. See https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values",
    )
    parser.add_argument("--cv-workers", type=int, default=4, help="Number of CV jobs to run in parallel.")
    parser.add_argument("--fx-workers", type=int, default=4, help="Number of processes to use for FX.")
    parser.add_argument(
        "--out-parquet-file",
        type=Path,
        required=True,
        help="Path to the output parquet file containing training info.",
    )
    parser.add_argument(
        "--out-models-dir",
        type=Path,
        required=True,
        help="Path to the dir where each model will be saved.",
    )
    args = parser.parse_args()

    main(args)
