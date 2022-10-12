from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


class DirConfig:
    def __init__(self, env: str, exp: str, debug: bool):
        self.root_dir = self._get_root_dir(env)
        self.input_dir = self._get_input_dir(env)
        self.output_root = self._get_output_dir(env, exp, debug)
        self.output_dir = self.output_root / "outputs"
        self.output_prediction_dir = self.output_root / "predictions"
        self.output_model_dir = self.output_root / "models"
        self.output_figure_dir = self.output_root / "figures"
        self.output_optimize_dir = self.output_root / "optimize"

        self._mkdir_if_not_exist(self.root_dir)
        self._mkdir_if_not_exist(self.input_dir)
        self._mkdir_if_not_exist(self.output_root)
        self._mkdir_if_not_exist(self.output_prediction_dir)
        self._mkdir_if_not_exist(self.output_model_dir)
        self._mkdir_if_not_exist(self.output_figure_dir)
        self._mkdir_if_not_exist(self.output_optimize_dir)

    def _mkdir_if_not_exist(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def _get_root_dir(self, env: str):
        if env == "colab":
            return Path("/content/coupon-allocation")
        return Path(__file__).parents[3]

    def _get_input_dir(self, env: str):
        if env == "colab":
            return Path("/content/drive/MyDrive/coupon-allocation/input")
        root_dir = self._get_root_dir(env)
        return root_dir / "input"

    def _get_output_dir(self, env: str, exp: str, debug: bool):
        if env == "colab":
            output_dir = Path("/content/drive/MyDrive/coupon-allocation/output") / exp
        elif env == "local":
            output_dir = self._get_root_dir(env) / "output" / exp

        if debug:
            return output_dir / "debug"
        else:
            return output_dir


@dataclass
class FeatureConfig:
    feature_names: List[str]


@dataclass
class PreprocessConfig:
    test_size: float = 0.2


@dataclass
class PostprocessConfig:
    n_sample: int = 10000
    n_seed: int = 10
    variant_no_to_coupon_type: Dict[int, str] = field(default_factory=dict)
    variant_no_to_cost: Dict[int, int] = field(default_factory=dict)


@dataclass
class Config:
    raw_dict: Dict
    name: str
    exp: str
    problem_type: str
    target_name: str
    treatment_name: str
    debug: bool
    seed: int
    env: str
    dir_config: DirConfig
    preprocess_config: PreprocessConfig
    postprocess_config: PostprocessConfig
    feature_config: FeatureConfig
