import sys
from pathlib import Path

import yaml
from src.services.base.base_service import BaseService
from src.services.config.config_interface import (
    Config,
    DirConfig,
    FeatureConfig,
    PreprocessConfig,
    PostprocessConfig
)


class ConfigService(BaseService):
    def __init__(self):
        super().__init__()

    def get_config(self) -> Config:
        return self.config

    def load_exp_config(self, exp: str, debug: bool = False):
        env = self._get_env()
        self.logger.info(f"env: {env}")
        dir_config = self._get_dir_config(env, exp, debug)
        for key, value in dir_config.__dict__.items():
            self.logger.info({key: value})

        yaml_dict = self._load_yaml(dir_config.root_dir / "yaml" / f"{exp}.yaml")

        preprocess_config = PreprocessConfig(**yaml_dict["preprocess_config"])
        postprocess_config = PostprocessConfig(**yaml_dict["postprocess_config"])
        feature_config = FeatureConfig(**yaml_dict["feature_config"])
        self.config = Config(
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
            preprocess_config=preprocess_config,
            postprocess_config=postprocess_config,
            feature_config=feature_config,
        )

        return self

    def _get_env(self) -> str:
        if "google.colab" in sys.modules:
            return "colab"
        return "local"

    def _get_dir_config(self, env: str, exp: str, debug: bool) -> DirConfig:
        return DirConfig(env, exp, debug)

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
    config_service.load_exp_config("test")
    print(config_service.config.postprocess_config)
