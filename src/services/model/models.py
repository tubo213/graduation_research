import copy
import pickle
from abc import abstractmethod

import numpy as np
import pandas as pd


class AbstractLearner:
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        raise NotImplementedError

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


class SLearner(AbstractLearner):
    def __init__(self, basemodel, control_no=0):
        self.basemodel = basemodel
        self.control_no = control_no

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        self.treatments = np.sort(treatment.unique())
        X["treatment"] = treatment
        self.basemodel.fit(X, y)

    def predict(self, X: pd.DataFrame):
        pred_df = pd.DataFrame()

        # control cvr
        control_X = X.copy()
        control_X["treatment"] = self.control_no
        control_pred = self.basemodel.predict(control_X)

        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue

            # treatment cvr
            treatment_X = X.copy()
            treatment_X["treatment"] = treatment_no
            treatment_pred = self.basemodel.predict(treatment_X)

            # treatment cvr - control cvr
            uplift = treatment_pred - control_pred
            pred_df[f"pred_{treatment_no}"] = uplift

        return pred_df


class TLearner(AbstractLearner):
    def __init__(self, basemodel, control_no=0):
        self.basemodel = copy.copy(basemodel)
        self.control_no = control_no
        self.treatment_basemodels = {}
        self.control_basemodel = copy.copy(basemodel)

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        self.treatments = np.sort(treatment.unique())
        control_idx = treatment == self.control_no
        control_X, control_y = X.loc[control_idx], y.loc[control_idx]
        self.control_basemodel.fit(control_X, control_y)

        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue

            treatment_idx = treatment == treatment_no
            treatment_X, treatment_y = X.loc[treatment_idx], y.loc[treatment_idx]
            treatment_basemodel_i = copy.copy(self.basemodel)
            treatment_basemodel_i.fit(treatment_X, treatment_y)
            self.treatment_basemodels[
                f"treatment_{treatment_no}"
            ] = treatment_basemodel_i

    def predict(self, X: pd.DataFrame):
        pred_df = pd.DataFrame()
        control_pred = self.control_basemodel.predict(X)

        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue

            treatment_pred = self.treatment_basemodels[
                f"treatment_{treatment_no}"
            ].predict(X)
            uplift = treatment_pred - control_pred
            pred_df[f"pred_{treatment_no}"] = uplift

        return pred_df


class XLearner(AbstractLearner):
    def __init__(self, basemodel, propensity_model, control_no=0):
        self.propensity_model = propensity_model
        self.basemodel = copy.copy(basemodel)
        self.control_no = control_no
        self.m0 = copy.copy(basemodel)
        self.m1s = {}
        self.d0s = {}
        self.d1s = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        control_idx = treatment == self.control_no
        control_X, control_y = X.loc[control_idx], y.loc[control_idx]
        self.treatments = np.sort(treatment.unique())
        self.propensity_model.fit(X, treatment)
        self.m0.fit(control_X, control_y)

        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue
            model_name = f"treatment_{treatment_no}"
            treatment_idx = treatment == treatment_no
            treatment_X, treatment_y = X.loc[treatment_idx], y.loc[treatment_idx]
            m1_i = copy.copy(self.basemodel)
            m1_i.fit(treatment_X, treatment_y)
            self.m1s[model_name] = m1_i

            tau0_y = m1_i.predict(control_X) - control_y
            tau1_y = treatment_y - self.m0.predict(treatment_X)
            d0_i = copy.copy(self.basemodel)
            d1_i = copy.copy(self.basemodel)
            d0_i.fit(control_X, tau0_y)
            d1_i.fit(treatment_X, tau1_y)
            self.d0s[model_name] = d0_i
            self.d1s[model_name] = d1_i

    def predict(self, X: pd.DataFrame):
        pred_df = pd.DataFrame()
        propensity_score = self.propensity_model.predict_proba(X)
        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue
            model_name = f"treatment_{treatment_no}"
            d0_pred = self.d0s[model_name].predict(X)
            d1_pred = self.d1s[model_name].predict(X)
            sum_propensity_score = propensity_score[:, [0, treatment_no]].sum(axis=1)
            uplift = (
                propensity_score[:, treatment_no] * d0_pred
                + propensity_score[:, treatment_no] * d1_pred
            ) / sum_propensity_score
            pred_df[f"pred_{treatment_no}"] = uplift

        return pred_df


class TransformedOutcomeLearner(AbstractLearner):
    def __init__(self, basemodel, control_no=0):
        self.basemodel = copy.copy(basemodel)
        self.basemodels = {}
        self.control_no = control_no

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        self.treatments = np.sort(treatment.unique())
        control_idx = treatment == self.control_no
        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue
            treatment_idx = treatment == treatment_no
            n_c = y.loc[control_idx].shape[0]
            n_t = y.loc[treatment_idx].shape[0]
            n = n_c + n_t
            control_X = X.loc[control_idx]
            control_y = -y.loc[control_idx] * (n / n_c)
            treatment_X = X.loc[treatment_idx]
            treatment_y = y.loc[treatment_idx] * (n / n_t)
            transformed_X = pd.concat([control_X, treatment_X])
            transformed_y = pd.concat([control_y, treatment_y])
            basemodel = copy.copy(self.basemodel)
            basemodel.fit(transformed_X, transformed_y)
            self.basemodels[f"treatment_{treatment_no}"] = basemodel

    def predict(self, X: pd.DataFrame):
        pred_df = pd.DataFrame()
        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue
            uplift = self.basemodels[f"treatment_{treatment_no}"].predict(X)
            pred_df[f"pred_{treatment_no}"] = uplift

        return pred_df


class CostConsiousTOLearner(AbstractLearner):
    def __init__(self, basemodel, clf_model, control_no=0):
        self.basemodel = copy.copy(basemodel)
        self.basemodels = {}
        self.control_no = control_no
        self.clf_model = clf_model

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series):
        self.treatments = np.sort(treatment.unique())
        control_idx = treatment == self.control_no
        control_positive_idx = (treatment == self.control_no) & (y > 0)
        # train prob control model
        control_X = X.loc[control_idx]
        control_prob = y.loc[control_idx].map(lambda x: 1 if x > 0 else 0)
        prob_control_model = copy.copy(self.clf_model)
        prob_control_model.fit(control_X, control_prob)
        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue
            treatment_idx = treatment == treatment_no
            treatment_positive_idx = (treatment == treatment_no) & (y > 0)
            treatment_X = X.loc[treatment_idx]
            treatment_prob = y.loc[treatment_idx].map(lambda x: 1 if x > 0 else 0)
            prob_treatment_model = copy.copy(self.clf_model)
            prob_treatment_model.fit(treatment_X, treatment_prob)
            # train uplift model
            n_c = y.loc[control_positive_idx].shape[0]
            n_t = y.loc[treatment_positive_idx].shape[0]
            n = n_c + n_t
            treatment_transformed_y = y.loc[treatment_positive_idx] * (n / n_t)
            treatment_positive_X = X.loc[treatment_positive_idx]
            control_positive_X = X.loc[control_positive_idx]
            prob_c_cp = prob_control_model.predict_proba(control_positive_X)[:, 1]
            prob_t_cp = prob_treatment_model.predict_proba(control_positive_X)[:, 1]
            w = prob_c_cp / prob_t_cp
            control_transformed_y = -w * y.loc[control_positive_idx] * (n / n_c)
            transformed_X = pd.concat([control_positive_X, treatment_positive_X])
            transformed_y = pd.concat([control_transformed_y, treatment_transformed_y])
            basemodel = copy.copy(self.basemodel)
            basemodel.fit(transformed_X, transformed_y)
            self.basemodels[f"treatment_{treatment_no}"] = basemodel

    def predict(self, X: pd.DataFrame):
        pred_df = pd.DataFrame()
        for treatment_no in self.treatments:
            if treatment_no == self.control_no:
                continue
            uplift = self.basemodels[f"treatment_{treatment_no}"].predict(X)
            pred_df[f"pred_{treatment_no}"] = uplift

        return pred_df
