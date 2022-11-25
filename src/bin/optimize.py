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

    # load uplift prediction
    uplift_pred = pd.read_csv(
        config.train_config.base_config.dir_config.output_prediction_dir / "uplift_pred.csv"
    ).to_numpy()

    # optimize
    optimizer = container.postprocess_service.get_postprocess()
    n_sample = min(len(test_df), postprocess_config.n_sample)
    for seed in tqdm(range(postprocess_config.n_seed), desc="seed"):
        np.random.seed(seed)
        sample_idx = test_df.sample(n_sample, random_state=seed).index.to_numpy()
        random_uplift_pred = np.random.rand(*uplift_pred.shape)
        for budget_constraint in tqdm(
            np.linspace(int(n_sample * 15 * 0.2), n_sample * 15, 10),
            desc="budget_constraint",
            leave=False,
        ):
            assginment = optimizer.postprocess(
                uplift_pred[sample_idx], budget_constraint, seed
            )
            np.save(
                config.base_config.dir_config.output_optimize_dir
                / f"assignment_{budget_constraint}_{seed}.npy",
                assginment,
            )
