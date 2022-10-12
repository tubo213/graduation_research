import numpy as np
import pandas as pd


def calc_auuc(cv, treatment, uplift_score):
    df = pd.DataFrame()
    df["cv_flg"] = cv.copy()
    df["treat_flg"] = treatment.copy()

    result = pd.DataFrame(
        np.c_[df["cv_flg"], df["treat_flg"], uplift_score.copy()],
        columns=["cv_flg", "treat_flg", "uplift_score"],
    )
    result = result.sort_values(by="uplift_score", ascending=False).reset_index(
        drop=True
    )

    result["treat_num_cumsum"] = result["treat_flg"].cumsum()
    result["control_num_cumsum"] = (1 - result["treat_flg"]).cumsum()
    result["treat_cv_cumsum"] = (result["treat_flg"] * result["cv_flg"]).cumsum()
    result["control_cv_cumsum"] = (
        (1 - result["treat_flg"]) * result["cv_flg"]
    ).cumsum()
    result["treat_cvr"] = (
        result["treat_cv_cumsum"] / result["treat_num_cumsum"]
    ).fillna(0)
    result["control_cvr"] = (
        result["control_cv_cumsum"] / result["control_num_cumsum"]
    ).fillna(0)
    result["lift"] = (result["treat_cvr"] - result["control_cvr"]) * result[
        "treat_num_cumsum"
    ]
    result["base_line"] = (
        result.index * result["lift"][len(result.index) - 1] / len(result.index)
    )

    auuc = (result["lift"] - result["base_line"]).sum() / len(result["lift"])

    return auuc


def plot_uplift_curve(cv, treatment, uplift_score, ax, model_name=None, baseline=True):
    df = pd.DataFrame()
    df["cv_flg"] = cv.copy()
    df["treat_flg"] = treatment.copy()

    result = pd.DataFrame(
        np.c_[df["cv_flg"], df["treat_flg"], uplift_score.copy()],
        columns=["cv_flg", "treat_flg", "uplift_score"],
    )
    result = result.sort_values(by="uplift_score", ascending=False).reset_index(
        drop=True
    )

    result["treat_num_cumsum"] = result["treat_flg"].cumsum()
    result["control_num_cumsum"] = (1 - result["treat_flg"]).cumsum()
    result["treat_cv_cumsum"] = (result["treat_flg"] * result["cv_flg"]).cumsum()
    result["control_cv_cumsum"] = (
        (1 - result["treat_flg"]) * result["cv_flg"]
    ).cumsum()
    result["treat_cvr"] = (
        result["treat_cv_cumsum"] / result["treat_num_cumsum"]
    ).fillna(0)
    result["control_cvr"] = (
        result["control_cv_cumsum"] / result["control_num_cumsum"]
    ).fillna(0)
    result["lift"] = (result["treat_cvr"] - result["control_cvr"]) * result[
        "treat_num_cumsum"
    ]
    result["base_line"] = (
        result.index * result["lift"][len(result.index) - 1] / len(result.index)
    )

    ax.plot(
        result.index / result.index[-1], result["lift"], label=model_name, linewidth=1
    )

    if baseline:
        ax.plot(
            result.index / result.index[-1],
            result["base_line"],
            label="baseline",
            color="black",
            linewidth=1,
        )
    ax.set_xlabel("percentage of total number of customers")
    ax.set_ylabel("cumulative uplift")
    ax.legend()

    return ax
