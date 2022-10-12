import matplotlib.pyplot as plt
import numpy as np
from argtyped import Arguments
from lightgbm import LGBMClassifier, LGBMRegressor
from src.containers.train_runner_container import TrainRunnerContainer
from src.model.metalearner import SLearner, TLearner, XLearner


class Args(Arguments):
    exp: str
    debug: bool = False


if __name__ == "__main__":
    args = Args()
    train_runner_container = TrainRunnerContainer(args.exp, args.debug)
    train_runner_container.initialize()
    config = train_runner_container.config_service.config

    train_df, test_df = train_runner_container.preprocess_service.split()

    # TODO:
    # - モデルの選択をconfigで指定できるようにする.
    # - モデルは1個ずつ学習するようにする.
    # - basemodelのクラスを作る. 今はsklearnAPIを使っているが, それをラップするクラスを作る.
    s_basemodel = LGBMRegressor(n_estimators=500)
    s_model = SLearner(s_basemodel)
    t_basemodel = LGBMRegressor(n_estimators=500)
    t_model = TLearner(t_basemodel)
    x_basemodel = LGBMRegressor(n_estimators=500)
    x_propensity_model = LGBMClassifier(n_estimators=500)
    x_model = XLearner(x_basemodel, x_propensity_model)
    feature_config = train_runner_container.config.feature_config

    # training
    s_model.fit(
        train_df[feature_config.feature_names],
        train_df[config.target_name],
        train_df[config.treatment_name],
    )
    t_model.fit(
        train_df[feature_config.feature_names],
        train_df[config.target_name],
        train_df[config.treatment_name],
    )
    x_model.fit(
        train_df[feature_config.feature_names],
        train_df[config.target_name],
        train_df[config.treatment_name],
    )

    # save models
    s_model.save(config.dir_config.output_model_dir / "s_model.pkl")
    t_model.save(config.dir_config.output_model_dir / "t_model.pkl")
    x_model.save(config.dir_config.output_model_dir / "x_model.pkl")

    # predict
    s_uplift_pred = s_model.predict(test_df[feature_config.feature_names])
    t_uplift_pred = t_model.predict(test_df[feature_config.feature_names])
    x_uplift_pred = x_model.predict(test_df[feature_config.feature_names])

    # save
    s_uplift_pred.to_csv(
        config.dir_config.output_prediction_dir / "s_uplift_pred.csv", index=False
    )
    t_uplift_pred.to_csv(
        config.dir_config.output_prediction_dir / "t_uplift_pred.csv", index=False
    )
    x_uplift_pred.to_csv(
        config.dir_config.output_prediction_dir / "x_uplift_pred.csv", index=False
    )

    # TODO: evaluationの結果をwandbに飛ばす. 今はローカルに保存するだけ
