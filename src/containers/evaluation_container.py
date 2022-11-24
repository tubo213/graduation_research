from ds_util import FileUtil, GlobalUtil
from src.services.base.base_service import BaseService
from src.services.config.config_service import ConfigService
from src.services.data.data_service import DataService
from src.services.preprocess.preprocess_service import PreprocessService


class EvaluationContainer(BaseService):
    def __init__(self, exp: str, debug: bool = False):
        super().__init__()
        self.config_service = ConfigService()
        self.config_service.load_exp_config(exp, debug)
        self.config = self.config_service.get_config()

    def initialize(self):
        GlobalUtil.seed_everything(self.config.base_config.seed)
        self.data_service = DataService(self.config_service)
        self.preprocess_service = PreprocessService(
            self.config, self.data_service
        )
        self.preprocess_service.preprocess()
        _, self.test_df = self.preprocess_service.split()

        self.uplift_preds = {
            "s_uplift_preds": FileUtil.load_csv(
                self.config.dir_config.output_prediction_dir / "s_uplift_pred.csv",
                logger=self.logger,
            ),
            "t_uplift_preds": FileUtil.load_csv(
                self.config.dir_config.output_prediction_dir / "t_uplift_pred.csv",
                logger=self.logger,
            ),
            "x_uplift_preds": FileUtil.load_csv(
                self.config.dir_config.output_prediction_dir / "x_uplift_pred.csv",
                logger=self.logger,
            ),
        }
        self.assignments = []
        output_optimize_paths = list(
            self.config.dir_config.output_optimize_dir.glob("*.npy")
        )
        for output_optimize_path in output_optimize_paths:
            assignment = FileUtil.load_npy(output_optimize_path, logger=self.logger)
            name, budget_constraint, seed = output_optimize_path.stem.split("_")
            seed = int(seed)
            budget_constraint = float(budget_constraint)
            self.assignments.append([name, seed, budget_constraint, assignment])
