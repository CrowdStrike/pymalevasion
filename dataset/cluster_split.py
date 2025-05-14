from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn import metrics
from tqdm.auto import tqdm

pd.set_option("display.max_colwidth", 100)
tqdm.pandas()


def make_fv_subset(fv_dict: dict[str, Any], keys: list[str]) -> list:
    """
    Construct a 1-dim FV by concatenating all sub-FVs in the FV dict given by keys.
    """
    out = []

    for k in keys:
        fv = fv_dict[k]
        [out.append, out.extend][isinstance(fv, (list, np.ndarray))](fv)

    return out


def hom(xs):
    # Homogeneity of the cluster
    cs = [(xs == "clean").sum(), (xs == "dirty").sum()]
    if 0 in cs:
        return 1

    return np.abs(cs[0] - cs[1]) / sum(cs)


if __name__ == "__main__":
    # To obtain the parquet with the structural feature vectors (FVs), use `do_fx.py`.
    df = pd.merge(
        left=pd.read_csv("hashes.csv"),
        right=pd.read_parquet("with_structural_fv.parquet"),
        on="sha256",
        how="inner",
    )
    assert df["parseable"].all()

    fvs = np.vstack(df["fv_dict"].apply(lambda d: make_fv_subset(d, keys=["fx_ast_node_count"])))
    print(fvs.shape)

    fvs_red = umap.UMAP(**{"metric": "hamming", "min_dist": 0.1, "n_components": 8, "n_neighbors": 200}).fit_transform(
        fvs
    )

    clus = hdbscan.HDBSCAN(
        **{
            "cluster_selection_epsilon": 1.0,
            "min_cluster_size": 100,
            "min_samples": 100,
            "cluster_selection_method": "eom",
        }
    ).fit_predict(fvs_red)

    df["cluster"] = clus

    print(df.groupby("cluster")["sha256"].agg(list).apply(len).max())

    print(metrics.calinski_harabasz_score(fvs_red, clus))
    print(metrics.silhouette_score(fvs_red, clus))

    df["int_label"] = df["label"].apply(lambda l: {"clean": 0, "dirty": 1}[l])

    cluster_sizes = df.groupby("cluster")["sha256"].agg(list).apply(len).sort_values(ascending=False).to_dict()

    cluster_mean_label = df.groupby("cluster")["int_label"].apply("mean").sort_values().to_dict()

    centroids = {}
    for i in df["cluster"].unique():
        centroids[i] = fvs_red[clus == i].mean(axis=0)

    pairwise_dist = {i: {j: np.linalg.norm(centroids[i] - centroids[j]) for j in centroids} for i in centroids}

    # this is what we want
    dirty_ratio = 0.5
    train_size = int(0.7 * len(df))
    valid_size = int(0.1 * len(df))
    test_size = len(df) - train_size - valid_size

    def choose(cs: set[int]):
        x = df.query("cluster.isin(@cs)")

        dirty = (x["label"] == "dirty").mean()
        size = len(x) / len(df)

        return dirty, size

    xs = set(cluster_mean_label)

    largest = max(cluster_sizes, key=cluster_sizes.get)
    train_clusters = {largest} | {k for k in xs if cluster_mean_label[k] >= 0.985}
    xs -= train_clusters
    print("train", choose(train_clusters))

    test_clusters = {k for k in xs if cluster_mean_label[k] <= 0.01} | {k for k in xs if cluster_mean_label[k] >= 0.9}
    xs -= test_clusters
    print("test", choose(test_clusters))

    valid_clusters = xs
    print("valid", choose(valid_clusters))

    new_df = df.copy()

    new_df.loc[new_df["cluster"].isin(train_clusters), "subset"] = "train"
    new_df.loc[new_df["cluster"].isin(test_clusters), "subset"] = "test"
    new_df.loc[new_df["cluster"].isin(valid_clusters), "subset"] = "valid"

    new_df.drop("fv_dict", axis=1).to_csv("cluster_split.csv", index=False, quoting=1)
