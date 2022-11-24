from typing import Union

import pandas as pd
from ds_util import FileUtil
from src.typing import CONFIGTYPE
from src.services.base.base_service import BaseService
from src.services.config.config_interface import OptimizeConfig, TrainConfig


class DataService(BaseService):
    def __init__(self, config: CONFIGTYPE) -> None:
        super().__init__()

        self.df: pd.DataFrame = FileUtil.load_csv(
            config.base_config.dir_config.input_dir
            / "retention_campaign_users_and_features_202202.csv",
            logger=self.logger,
        )
        if config.base_config.debug:
            self.df = self.df.sample(500).reset_index(drop=True)

        self.logger.info(f"df.shape: {self.df.shape}")
