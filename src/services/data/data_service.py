import pandas as pd
from ds_util import FileUtil
from src.services.base.base_service import BaseService
from src.services.config.config_service import ConfigService


class DataService(BaseService):
    def __init__(self, config_service: ConfigService):
        super().__init__()
        config = config_service.get_config()

        self.df: pd.DataFrame = FileUtil.load_csv(
            config.dir_config.input_dir
            / "retention_campaign_users_and_features_202202.csv",
            logger=self.logger,
        )
        if config.debug:
            self.df = self.df.sample(1000).reset_index(drop=True)

        self.logger.info(f"df.shape: {self.df.shape}")
