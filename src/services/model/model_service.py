import multiprocessing
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import catboost as cat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
from src.services.base.base_service import BaseService
from src.services.model.models import AbstractLearner, SLearner, TLearner, XLearner, TransformedOutcomeLearner, CostConsiousTOLearner
from src.typing import CONFIGTYPE


@dataclass
class ModelService(BaseService):
    def __init__(self, config: CONFIGTYPE):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config

    def get_model(self) -> AbstractLearner:
        basemodel = self._get_basemodel(self.config.base_config.problem_type)

        if self.model_config.metalearner["name"] == "s-learner":
            return SLearner(basemodel, **self.model_config.metalearner["params"])
        elif self.model_config.metalearner["name"] == "t-learner":
            return TLearner(basemodel, **self.model_config.metalearner["params"])
        elif self.model_config.metalearner["name"] == "x-learner":
            propensity_model = self._get_clf_basemodel()
            return XLearner(
                basemodel,
                propensity_model=propensity_model,
                **self.model_config.metalearner["params"],
            )
        elif self.model_config.metalearner["name"] == "transformedoutcome":
            return TransformedOutcomeLearner(
                basemodel, **self.model_config.metalearner["params"]
            )
        elif self.model_config.metalearner["name"] == "costconsious":
            clf_model = self._get_clf_basemodel()
            return CostConsiousTOLearner(
                basemodel, clf_model, **self.model_config.metalearner["params"]
            )
        else:
            raise ValueError(
                f"""
                invalid metalearner name: {self.model_config.metalearner['name']},
                metalearner name must be 's-learner' or 't-learner' or 'x-learner' or 'transformedoutcome' or 'costconsious'
                """
            )

    def _get_basemodel(self, problem_type):
        if problem_type == "regression":
            return self._get_reg_basemodel()
        elif problem_type == "classification":
            return self._get_clf_basemodel()
        else:
            raise ValueError(
                f"""
                invalid problem_type: {problem_type},
                problem_type must be 'regression' or 'classification'
                """
            )

    def _get_reg_basemodel(self):
        if self.model_config.basemodel['name'] == "lgbm":
            return lgb.LGBMRegressor(**self.model_config.basemodel['params'])
        elif self.model_config.basemodel['name'] == "xgb":
            return xgb.XGBRegressor(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "catboost":
            return cat.CatBoostRegressor(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "rf":
            return RandomForestRegressor(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "mlp":
            return MLPRegressor(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "linear":
            return LinearRegression(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "ridge":
            return Ridge(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "lasso":
            return Lasso(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "elasticnet":
            return ElasticNet(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "svr":
            return SVR(**self.model_config.basemodel["params"])
        else:
            raise ValueError(
                f"""
                invalid basemodel name: {self.model_config.basemodel.name},
                basemodel name must be 'lgbm' or 'xgb' or 'catboost' or 'rf' or 'mlp' or 'linear' or 'ridge' or 'lasso' or 'elasticnet' or 'svr'
                """
            )

    def _get_clf_basemodel(self):
        if self.model_config.basemodel['name'] == "lgbm":
            return lgb.LGBMClassifier(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "xgb":
            return xgb.XGBClassifier(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "catboost":
            return cat.CatBoostClassifier(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "rf":
            return RandomForestClassifier(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "mlp":
            return MLPClassifier(**self.model_config.basemodel["params"])
        elif self.model_config.basemodel['name'] == "svm":
            return SVC(**self.model_config.basemodel["params"])
        else:
            raise ValueError(
                f"""
                invalid basemodel name: {self.model_config.basemodel['name']},
                basemodel name must be 'lgbm' or 'xgb' or 'catboost' or 'rf' or 'mlp' or 'logistic' or 'ridge' or 'svm'
                """
            )
