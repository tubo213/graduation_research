import matplotlib.pyplot as plt
import numpy as np
from argtyped import Arguments
from src.containers.train_runner_container import TrainRunnerContainer
from src.visuzalize.visualize_uplift import calc_auuc, plot_uplift_curve


class Args(Arguments):
    exp: str
    debug: bool = False


if __name__ == "__main__":
    args = Args()
    train_runner_container = TrainRunnerContainer(args.exp, args.debug)
    train_runner_container.initialize()
    config = train_runner_container.config_service.config

    train_df, test_df = train_runner_container.preprocess_service.split()

    model = train_runner_container.model_service.get_model()
    feature_config = train_runner_container.config.feature_config

    # training
    train_runner_container.logger.info("start training")
    model.fit(
        train_df[feature_config.feature_names],
        train_df[config.base_config.target_name],
        train_df[config.base_config.treatment_name],
    )
    train_runner_container.logger.info("end training")

    # save models
    model.save(config.base_config.dir_config.output_model_dir / "model.pkl")

    # predict
    train_runner_container.logger.info("start predict")
    uplift_pred = model.predict(test_df[feature_config.feature_names])
    train_runner_container.logger.info("end predict")

    # save
    uplift_pred.to_csv(
        config.base_config.dir_config.output_prediction_dir / "uplift_pred.csv",
        index=False,
    )

    train_runner_container.logger.info("end")

    # evaluate
    train_runner_container.logger.info("start evaluate")
    treatment_no_list = np.sort(
        test_df[config.base_config.treatment_name].unique()
    ).tolist()
    treatment_no_list.remove(0)
    for i, treatment_no in enumerate(treatment_no_list):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        uplift_col = f"pred_{treatment_no}"
        coupon_type = config.base_config.counpon_config.variant_no_to_coupon_type[
            treatment_no
        ]

        test_index_i = test_df.query("treatment in [0, @treatment_no]").index
        test_df_i = test_df.loc[test_index_i]
        preds_i = uplift_pred.loc[test_index_i, uplift_col]
        treatment_i = test_df_i[config.base_config.treatment_name] == treatment_no
        if config.base_config.problem_type == "classification":
            target_i = test_df_i[config.base_config.target_name] > 0
        else:
            target_i = test_df_i[config.base_config.target_name]

        # plot uplift curve
        plot_uplift_curve(
            target_i,
            treatment_i,
            preds_i,
            ax=ax,
            model_name=config.model_config.metalearner["name"],
            baseline=True,
        )
        ax.set_title(f"uplift curve: {coupon_type}")
        auuc = calc_auuc(target_i, treatment_i, preds_i)
        train_runner_container.logger.info(f"{coupon_type} AUUC: {auuc}")
        save_path = str(
            config.base_config.dir_config.output_figure_dir / "uplift_curve.png"
        )
        fig.savefig(save_path)
