import sys
from pathlib import Path
from typing import Union, Tuple

import yaml
from src.typing import CONFIGTYPE
from src.services.base.base_service import BaseService
from src.services.config.config_interface import (
    BaseConfig,
    DirConfig,
    FeatureConfig,
    ModelConfig,
    OptimizeConfig,
    PostprocessConfig,
    PreprocessConfig,
    TrainConfig,
    CouponConfig
)


class ConfigService(BaseService):
    def __init__(self):
        super().__init__()

    def get_config(self) -> CONFIGTYPE:
        return self.config

    def load_exp_config(self, exp: str, debug: bool = False):
        base_config, yaml_dict = self._get_base_config_and_yaml_dict(exp, debug)

        if exp.startswith("train"):
            self.config = self._get_train_config(base_config, yaml_dict)
        elif exp.startswith("optimize"):
            self.config = self._get_optimize_config(base_config, yaml_dict)
        else:
            raise ValueError(f"invalid exp name: {exp}")

        return self

    def _get_base_config_and_yaml_dict(self, exp: str, debug: bool = False, verbose: bool = True) -> Tuple[BaseConfig, dict]:
        env = self._get_env()
        dir_config = self._get_dir_config(env, exp, debug)
        for key, value in dir_config.__dict__.items():
            self.logger.info({key: value})

        yaml_dict = self._load_yaml(dir_config.root_dir / "yaml" / f"{exp}.yaml", verbose)
        base_config = BaseConfig(
            raw_dict=yaml_dict,
            name=yaml_dict["name"],
            exp=exp,
            debug=debug,
            seed=int(yaml_dict["seed"]),
            target_name=yaml_dict["target_name"],
            treatment_name=yaml_dict["treatment_name"],
            problem_type=yaml_dict["problem_type"],
            env=env,
            dir_config=dir_config,
            counpon_config=CouponConfig(**yaml_dict["coupon_config"])
        )
        return base_config, yaml_dict

    def _get_train_config(
        self, base_config: BaseConfig, yaml_dict: dict
    ) -> TrainConfig:
        train_config = TrainConfig(
            base_config=base_config,
            preprocess_config=PreprocessConfig(**yaml_dict["preprocess_config"]),
            feature_config=FeatureConfig(**yaml_dict["feature_config"]),
            model_config=ModelConfig(**yaml_dict["model_config"]),
        )

        return train_config

    def _get_optimize_config(
        self, base_config: BaseConfig, yaml_dict: dict
    ) -> OptimizeConfig:
        train_base_config, train_yaml_dict = self._get_base_config_and_yaml_dict(
            yaml_dict["train_name"], base_config.debug, verbose=False
        )
        train_config = self._get_train_config(train_base_config, train_yaml_dict)
        optimize_config = OptimizeConfig(
            base_config=base_config,
            train_config=train_config,
            postprocess_config=PostprocessConfig(**yaml_dict["postprocess_config"]),
        )

        return optimize_config

    def _get_dir_config(self, env: str, exp: str, debug: bool) -> DirConfig:
        return DirConfig(env, exp, debug)

    def _get_feature_config(self, yaml_dict: dict) -> FeatureConfig:
        return FeatureConfig(**yaml_dict["feature_config"])

    def _get_model_config(self, yaml_dict: dict) -> ModelConfig:
        return ModelConfig(**yaml_dict["model_config"])

    def _get_preprocess_config(self, yaml_dict: dict) -> PreprocessConfig:
        return PreprocessConfig(**yaml_dict["preprocess_config"])

    def _get_postprocess_config(self, yaml_dict: dict) -> PostprocessConfig:
        return PostprocessConfig(**yaml_dict["postprocess_config"])

    def _get_env(self) -> str:
        if "google.colab" in sys.modules:
            return "colab"
        return "local"

    def _load_yaml(self, file_path: Path, verbose: bool = True) -> dict:
        self.logger.info(f"load yaml file: {file_path}")
        with open(file_path, "rb") as f:
            yaml_dict = yaml.safe_load(f)
        if verbose:
            with open(file_path, "r") as f:
                print(f"=============== {file_path} =================")
                print(f.read())
                print("==============================================")
        return yaml_dict


if __name__ == "__main__":
    config_service = ConfigService()
    config_service.load_exp_config("train001")
    config_service.load_exp_config("optimize001")
