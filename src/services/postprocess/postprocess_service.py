from dataclasses import dataclass
from typing import Union

from src.services.base.base_service import BaseService
from src.services.data.data_service import DataService
from src.services.postprocess.cluster_postprocess import ClusterPostprocess
from src.services.postprocess.cluster_with_uncertainty_postprocess import (
    ClusterWithUncertaintyPostprocess,
)
from src.services.postprocess.discrete_postprocess import DiscretePostprocess
from src.services.postprocess.greedy_postprocess import GreedyPostprocess
from src.typing import CONFIGTYPE

POSTPROCESSTYPE = Union[
    ClusterPostprocess,
    ClusterWithUncertaintyPostprocess,
    DiscretePostprocess,
    GreedyPostprocess,
]


@dataclass
class PostprocessService(BaseService):
    def __init__(
        self, config: CONFIGTYPE, data_service: DataService
    ) -> POSTPROCESSTYPE:
        super().__init__()
        self.config = config
        self.postprocess_config = self.config.postprocess_config
        self.data_service = data_service

    def get_postprocess(self):
        if self.postprocess_config.name == "discrete":
            return DiscretePostprocess(self.postprocess_config)
        elif self.postprocess_config.name == "cluster":
            return ClusterPostprocess(self.postprocess_config)
        elif self.postprocess_config.name == "cluster_with_uncertainty":
            return ClusterWithUncertaintyPostprocess(self.postprocess_config)
        elif self.postprocess_config.name == "greedy":
            return GreedyPostprocess(self.postprocess_config)
        else:
            raise ValueError(
                f"""
                invalid postprocess name: {self.postprocess_config.name},
                postprocess name must be 'discrete' or 'cluster' or 'cluster_with_uncertainty' or 'greedy'
                """
            )
