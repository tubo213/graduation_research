import math

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.services.base.base_service import BaseService
from src.utils import save_as_pickle, load_from_pickle


class ClusterWithUncertaintyPostprocess(BaseService):
    def __init__(self, base_config, postprocess_config):
        super().__init__()
        self.base_config = base_config
        self.postprocess_config = postprocess_config
        self.cost = np.array(
            list(self.postprocess_config.variant_no_to_cost.values())[1:]
        )
        self.n_cluster = self.postprocess_config.params["n_cluster"]

    def postprocess(self, uplift_mat, budget_constraint: float, seed: int):
        # clustering
        self.n_treatment = uplift_mat.shape[1]
        labels = self.run_clustering(uplift_mat, seed)

        uplift_df = pd.DataFrame(uplift_mat)
        pi_bar = uplift_df.groupby(labels).mean().sort_index().values.flatten()
        cov_mat = self.get_cov_mat(uplift_df, labels)

        # optimize
        x = self.optimize(
            pi_bar, cov_mat, labels, budget_constraint
        )  # x.shape = (n_cluster, n_treatment)

        # sampling each cluster
        assign_df = pd.DataFrame(index=uplift_df.index, columns=["assign"])
        assign_df["assign"] = 0
        assign_df["cluster"] = labels

        for cluster in range(self.n_cluster):
            for coupon in range(self.n_treatment):
                _assign_df = assign_df.query(f"cluster == {cluster} and assign == 0")
                n_assign = min(math.floor(x[cluster, coupon]), _assign_df.shape[0])
                if n_assign > 0:
                    _sample_idx = _assign_df.sample(n_assign, random_state=0).index
                    assign_df.loc[_sample_idx, "assign"] = coupon + 1

        return assign_df["assign"].to_numpy()

    def get_clustering(self):
        if self.postprocess_config.params["clustering"] == "kmeans":
            return KMeans(
                n_clusters=self.n_cluster,
                random_state=self.base_config.seed,
            )
        elif self.postprocess_config.params["clustering"] == "gmm":
            return GaussianMixture(
                n_components=self.n_cluster,
                random_state=self.base_config.seed,
            )

    def run_clustering(self, uplift_mat, seed: int):
        model_path = self.base_config.dir_config.output_model_dir / f"clustering_model_{seed}.pkl"
        label_path = self.base_config.dir_config.output_optimize_dir / f"cluster_{seed}.npy"
        if model_path.exists():
            clustering_model = load_from_pickle(model_path)
            labels = np.load(label_path)
        else:
            clustering_model = self.get_clustering()
            clustering_model.fit(uplift_mat)
            save_as_pickle(clustering_model, model_path)
            labels = clustering_model.predict(uplift_mat)
            np.save(label_path, labels)

        return labels

    def get_cov_mat(self, uplift_df: pd.DataFrame, labels):
        bootstrap_samples = []
        B = 1000
        for i in range(B):
            sample_idx = np.random.choice(len(uplift_df), len(uplift_df), replace=True)
            sample_uplift_pred = uplift_df.iloc[sample_idx]
            sample_uplift_pred = (
                sample_uplift_pred.groupby(labels).mean().sort_index().values.flatten()
            )
            bootstrap_samples.append(sample_uplift_pred)
        bootstrap_samples = np.stack(bootstrap_samples)
        cov_mat = np.cov(bootstrap_samples, rowvar=False, bias=False)

        return cov_mat

    def optimize(self, pi_bar, cov_mat, labels, budget_constraint):
        gamma = cp.Variable(shape=(self.n_cluster * self.n_treatment, 1), nonneg=True)
        cov_mat = cp.atoms.affine.wraps.psd_wrap(cov_mat)

        # Objective (Return & Risk)
        ret = pi_bar @ gamma
        risk = cp.quad_form(gamma, cov_mat)
        lam = self.postprocess_config.params["lambda"]
        objective = cp.Maximize((1 - lam) * ret - lam * risk)

        # Constraints
        constraints = []

        # # Budget Constraint
        costs = np.tile(self.cost, self.n_cluster)
        constraints.append(costs @ gamma <= budget_constraint)

        # # Cluster size Constraint
        cluster_sizes = np.unique(labels, return_counts=True)[1]
        for i in range(cluster_sizes.shape[0]):
            size = cluster_sizes[i]
            _gamma = gamma[i * self.n_treatment : (i + 1) * self.n_treatment]
            constraints.append(cp.sum(_gamma) <= size)

        # solve
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        return gamma.value.reshape(self.n_cluster, self.n_treatment)
