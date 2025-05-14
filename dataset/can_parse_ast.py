"""
Checks if Python files can be parsed into ASTs.
"""

import ast
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def work(f):
    try:
        with open(f, "rt") as fp:
            t = ast.parse(fp.read())
    except Exception as e:
        t = None

    return f.stem, (t is not None)


if __name__ == "__main__":
    df = []
    files = list(Path("./data/").iterdir())

    with mp.Pool(15) as pool:
        with tqdm(total=len(files)) as pbar:
            for h, res in pool.imap_unordered(work, files):
                df.append({"sha256": h, "ast": res})
                pbar.update(1)

    df = pd.DataFrame(df)
    df.to_csv("./parseable.csv", index=False, quoting=1)
