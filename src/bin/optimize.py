import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argtyped import Arguments
from src.containers.optimize_runner_container import OptimizeRunnerContainer
from tqdm import tqdm


class Args(Arguments):
    exp: str
    debug: bool = False


if __name__ == "__main__":
    args = Args()
    container = OptimizeRunnerContainer(args.exp, args.debug)
    container.initialize()
    config = container.config_service.config
    postprocess_config = config.postprocess_config

    train_df, test_df = container.preprocess_service.split()

    # load uplift predictions
    # TODO: 最適化用のデータ読み込みクラス作る
    s_uplift_pred = pd.read_csv(config.dir_config.output_prediction_dir / "s_uplift_pred.csv")
    t_uplift_pred = pd.read_csv(config.dir_config.output_prediction_dir / "t_uplift_pred.csv")
    x_uplift_pred = pd.read_csv(config.dir_config.output_prediction_dir / "x_uplift_pred.csv")
    pred_names = ["s-learner", "t-learner", "x-learner"]

    # optimize
    # TODO: configで最適化の方法を指定できるようにする
    n_sample = min(len(test_df), postprocess_config.n_sample)
    for seed in tqdm(range(postprocess_config.n_seed), desc="seed"):
        np.random.seed(seed)
        sample_idx = test_df.sample(n_sample, random_state=seed).index.to_numpy()
        random_uplift_pred = np.random.rand(*s_uplift_pred.shape)
        uplift_preds = [
            s_uplift_pred.to_numpy(),
            t_uplift_pred.to_numpy(),
            x_uplift_pred.to_numpy(),
        ]
        for budget_constraint in tqdm(
            np.linspace(int(n_sample * 15 * 0.2), n_sample * 15, 10),
            desc="budget_constraint",
            leave=False,
        ):
            for name, preds in zip(pred_names, uplift_preds):
                assginment = container.postprocess_service.postprocess(
                    preds[sample_idx], budget_constraint
                )
                np.save(
                    config.dir_config.output_optimize_dir
                    / f"{name}_{budget_constraint}_{seed}.npy",
                    assginment,
                )
