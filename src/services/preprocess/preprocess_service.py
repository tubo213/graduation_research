from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from src.services.base.base_service import BaseService
from src.services.config.config_service import ConfigService
from src.services.data.data_service import DataService


@dataclass
class PreprocessService(BaseService):
    def __init__(self, config_service: ConfigService, data_service: DataService):
        super().__init__()
        self.config_service = config_service
        self.data_service = data_service

    def preprocess(self):
        df = self.data_service.df
        df["treatment"] = df["variant_no"]
        self.data_service.df = df

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            self.data_service.df,
            test_size=self.config_service.config.preprocess_config.test_size,
            random_state=self.config_service.config.seed,
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
