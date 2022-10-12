from ds_util import GlobalUtil
from src.services.base.base_service import BaseService
from src.services.config.config_service import ConfigService
from src.services.data.data_service import DataService
from src.services.preprocess.preprocess_service import PreprocessService


class TrainRunnerContainer(BaseService):
    def __init__(self, exp: str, debug: bool = False):
        super().__init__()
        self.config_service = ConfigService()
        self.config_service.load_exp_config(exp, debug)
        self.config = self.config_service.get_config()

    def initialize(self):
        GlobalUtil.seed_everything(self.config.seed)
        self.data_service = DataService(self.config_service)
        self.preprocess_service = PreprocessService(self.config_service, self.data_service)
        self.preprocess_service.preprocess()
