from ds_util import FileUtil, GlobalUtil
from src.services.base.base_service import BaseService
from src.services.config.config_service import ConfigService
from src.services.data.data_service import DataService
from src.services.preprocess.preprocess_service import PreprocessService


class EvaluationContainer(BaseService):
    def __init__(self, exp: str, debug: bool = False):
        super().__init__()
        self.load(exp, debug)

    def load(self, exp: str, debug: bool):
        if exp.startswith("train"):
            self.bin = "train"
        elif exp.startswith("optimize"):
            self.bin = "optimize"

        self.config_service = ConfigService()
        self.config_service.load_exp_config(exp, debug)
        self.config = self.config_service.get_config()

    def initialize(self):
        GlobalUtil.seed_everything(self.config.base_config.seed)
        self.data_service = DataService(self.config)

        if self.bin == "train":
            self.preprocess_service = PreprocessService(self.config, self.data_service)
        elif self.bin == "optimize":
            self.preprocess_service = PreprocessService(
                self.config.train_config, self.data_service
            )

        self.preprocess_service.preprocess()
        _, self.test_df = self.preprocess_service.split()

        self.uplift_pred = self.load_uplift()

        if self.bin == "optimize":
            self.assignments = self.load_assignments()
            self.clusters = self.load_clusters()

    def load_uplift(self):
        if self.bin == "train":
            path = (
                self.config.base_config.dir_config.output_prediction_dir
                / "uplift_pred.csv"
            )
        elif self.bin == "optimize":
            path = (
                self.config.train_config.base_config.dir_config.output_prediction_dir
                / "uplift_pred.csv"
            )

        uplift_pred = FileUtil.load_csv(path)
        return uplift_pred

    def load_assignments(self):
        if self.bin == "train":
            raise ValueError("train bin does not have assignment")

        assignment_paths = list(
            self.config.base_config.dir_config.output_optimize_dir.glob(
                "assignment*.npy"
            )
        )
        assignments = []
        for path in assignment_paths:
            assignment = FileUtil.load_npy(path, logger=self.logger)
            _, budget_constraint, seed = path.stem.split("_")
            seed = int(seed)
            budget_constraint = float(budget_constraint)
            assignments.append([seed, budget_constraint, assignment])

        return assignments

    def load_clusters(self):
        if self.bin == "train":
            raise ValueError("train bin does not have assignment")

        cluster_paths = list(
            self.config.base_config.dir_config.output_optimize_dir.glob("cluster*.npy")
        )
        clusters = []
        for path in cluster_paths:
            assignment = FileUtil.load_npy(path, logger=self.logger)
            _, seed, budget_constraint = path.stem.split("_")
            seed = int(seed)
            budget_constraint = float(budget_constraint)
            clusters.append([seed, budget_constraint, assignment])

        return clusters
