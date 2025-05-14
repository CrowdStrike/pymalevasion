"""
Given results (run_id ...) and trained models, predict on a list of hashes.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shallow_fx
from loguru import logger
from tqdm import tqdm

ALL_FX_FUNCS = shallow_fx.collect_all_fxs()


def make_fv_dict(filepath: Path) -> dict[str, Any] | None:
    """
    The FV dict contains the mapping from FX name to FV from all FX functions.
    """

    try:
        with open(filepath, "rt") as fp:
            source = fp.read()
        fv = {fx: getattr(shallow_fx, fx)(source) for fx in ALL_FX_FUNCS}
    except (UnicodeDecodeError, SyntaxError):
        fv = None

    return filepath.name, fv


def make_fv_subset(fv_dict: dict[str, Any], keys: list[str]) -> list:
    """
    Construct a 1-dim FV by concatenating all sub-FVs in the FV dict given by keys.
    """
    out = []

    for k in keys:
        fv = fv_dict[k]
        [out.append, out.extend][isinstance(fv, list)](fv)

    return out


def main(args):
    df = pd.read_parquet(args.training_results_parquet)
    logger.info(f"Training results parquet: {df.shape}")

    # load csv with hashes to predict on
    to_pred = pd.read_csv(args.hashes_to_pred)
    col = args.column
    hashes_to_pred = set(to_pred[col])
    logger.info(f"{len(hashes_to_pred)=}")

    # collect paths of samples to predict on
    paths = set()
    for sdir in args.samples_dirs:
        for x in sdir.glob("**/*"):
            # a = x.stem
            # b = f"{x.stem}.py"
            # if (a in hashes_to_pred or b in hashes_to_pred):
            #     paths.add(x)
            paths.add(x)
    logger.info(f"Will predict on {len(paths)=} files")
    assert 0 < len(paths) <= len(hashes_to_pred), f"{len(paths)=}; {len(hashes_to_pred)=}"

    # construct feature vectors for all samples
    fv_dict = {}  # filename -> FV dict
    err = 0
    with mp.Pool(args.fx_workers) as pool:
        for name, res in tqdm(pool.imap_unordered(make_fv_dict, paths), total=len(paths), desc="FX"):
            if res is None:
                # print(f"Error for {name}")
                err += 1
                continue
            fv_dict[name] = res
    assert len(fv_dict) > 0
    logger.info(f"Constructed FVs for {len(fv_dict)} samples ({err} errors)")

    file2label = None
    if "label" in to_pred:
        file2label = {i[col]: {"clean": 0, "dirty": 1}[i["label"]] for _, i in to_pred[[col, "label"]].iterrows()}

    # predict with each trained model
    res = []
    for r in tqdm(df[["run_id", "fx_funcs"]].to_dict(orient="records"), desc="Pred"):
        cur = r

        model = joblib.load(args.models_dir / f"model_{r['run_id']}.sklearn")

        fvs = np.vstack([make_fv_subset(d, keys=r["fx_funcs"]) for d in fv_dict.values()])
        assert fvs.ndim == 2 and len(fvs) == len(fv_dict)

        cur["preds"] = model.predict_proba(fvs)[:, 1]
        cur["files"] = list(fv_dict.keys())

        if file2label is not None:
            cur["labels"] = [file2label[f] for f in cur["files"]]

        res.append(r)

    pd.DataFrame(res).to_parquet(args.out_parquet)
    logger.info(f"Done! Saved to {args.out_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-results-parquet", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--column", type=str, required=True)
    parser.add_argument("--samples-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--hashes-to-pred", type=Path, required=True, help="Subset into `samples-dirs`.")
    parser.add_argument("--fx-workers", type=int, default=4, help="Number of processes to use for FX.")
    parser.add_argument("--out-parquet", type=Path, required=True)
    args = parser.parse_args()

    main(args)
