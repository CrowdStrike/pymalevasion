import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm.auto import tqdm

tqdm.pandas()


def compute_optimal_f1_threshold(true, pred):
    f1_max = -1
    out_dv = None
    for dv in np.linspace(0.1, 0.9, endpoint=True, num=50):
        f1 = metrics.f1_score(true, pred >= dv)
        if f1 > f1_max:
            f1_max = f1
            out_dv = dv

    assert out_dv is not None
    return out_dv


def main():
    # To get preds use `shallow_predict.py`
    df = pd.read_parquet("preds.parquet")

    df["fx_count"] = df["fx_funcs"].apply(len)
    df["thr_f1"] = df.progress_apply(
        lambda row: compute_optimal_f1_threshold(row["valid_labels"], row["valid_preds"]), axis=1
    )
    df["eval_f1_05"] = df.apply(lambda row: metrics.f1_score(row["test_labels"], row["test_preds"] >= 0.5), axis=1)
    df["eval_precision_05"] = df.apply(
        lambda row: metrics.precision_score(row["test_labels"], row["test_preds"] >= 0.5), axis=1
    )
    df["eval_recall_05"] = df.apply(
        lambda row: metrics.recall_score(row["test_labels"], row["test_preds"] >= 0.5), axis=1
    )
    df["eval_accuracy_05"] = df.apply(
        lambda row: metrics.accuracy_score(row["test_labels"], row["test_preds"] >= 0.5), axis=1
    )
    df["eval_f1_custom"] = df.apply(
        lambda row: metrics.f1_score(row["test_labels"], row["test_preds"] >= row["thr_f1"]),
        axis=1,
    )
    df["eval_precision_custom"] = df.apply(
        lambda row: metrics.precision_score(row["test_labels"], row["test_preds"] >= row["thr_f1"]),
        axis=1,
    )
    df["eval_recall_custom"] = df.apply(
        lambda row: metrics.recall_score(row["test_labels"], row["test_preds"] >= row["thr_f1"]),
        axis=1,
    )
    df["eval_accuracy_custom"] = df.apply(
        lambda row: metrics.accuracy_score(row["test_labels"], row["test_preds"] >= row["thr_f1"]),
        axis=1,
    )
    df["eval_log_loss"] = df.apply(lambda row: metrics.log_loss(row["test_labels"], row["test_preds"]), axis=1)
    df["eval_auc"] = df.apply(lambda row: metrics.roc_auc_score(row["test_labels"], row["test_preds"]), axis=1)

    tmp = pd.concat(
        [df["run_id"], df["fx_funcs"], df["thr_f1"], df.filter(regex="eval_*"), df["fx_count"]], axis=1
    ).sort_values("run_id")
    tmp.to_excel("pred_view.xlsx")


if __name__ == "__main__":
    main()
