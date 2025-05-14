"""
Generate a parquet file with all FX combinations.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd
import shallow_fx
from loguru import logger

ALL_FX_FUNCS = shallow_fx.collect_all_fxs()


def make_fv_dict(filepath: str) -> dict[str, Any]:
    """
    The FV dict contains the mapping from FX name to FV from all FX functions.
    """
    with open(filepath, "rt") as fp:
        source = fp.read()

    return {fx: getattr(shallow_fx, fx)(source) for fx in ALL_FX_FUNCS}


def main(args) -> None:
    """
    Apply full FX on the input CSV and save the results (with extra `fv_dict` column) as parquet.
    """
    assert args.out_parquet_file.suffix == ".parquet"

    df = pd.read_csv(args.input_csv).query("parseable == True")

    if "path" in df.columns:
        logger.info("Found path column in dataframe")
        paths = df["path"].apply(Path)
    else:
        assert args.sources_dir is not None
        logger.info(f"No path column in dataframe, reading sources from {args.sources_dir}")
        paths = df["sha256"].apply(lambda h: args.sources_dir / h)
    assert all(p.exists() for p in paths)

    logger.info(f"Doing FX with {args.workers} workers on {len(paths)} samples...")
    with mp.Pool(args.workers) as pool:
        df["fv_dict"] = pool.map(make_fv_dict, paths)

    df[["sha256", "fv_dict"]].to_parquet(args.out_parquet_file)


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
    parser.add_argument("--workers", type=int, default=4, help="Number of processes to use for FX.")
    parser.add_argument(
        "--out-parquet-file",
        type=Path,
        required=True,
        help="Path to the output parquet file containing `sha256,fv_dict`",
    )
    args = parser.parse_args()

    main(args)
