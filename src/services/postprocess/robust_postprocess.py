import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.services.base.base_service import BaseService
from src.utils import save_as_pickle


class RobustPostprocess(BaseService):
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
        clustering_model = self.get_clustering()
        clustering_model.fit(uplift_mat)
        save_clustering_model_path = (
            self.base_config.dir_config.output_model_dir
            / f"clustering_model_{seed}_{budget_constraint}.pkl"
        )
        save_as_pickle(clustering_model, save_clustering_model_path)

        # # get cluster labels
        labels = clustering_model.predict(uplift_mat)
        np.save(
            self.base_config.dir_config.output_optimize_dir
            / f"cluster_{seed}_{budget_constraint}.npy",
            labels,
        )

        uplift_df = pd.DataFrame(uplift_mat)
        cluster_uplift = uplift_df.groupby(labels).mean().values.flatten()
        cluster_size = uplift_df.groupby(labels).size().values
        sigma = np.stack(
            [np.sqrt(np.diag(_cov)) for _cov in clustering_model.covariances_]
        ).flatten()
        sigma = self.postprocess_config.params['alpha'] * sigma
        costs = np.tile(self.cost, self.n_cluster).flatten()

        # optimize
        x = self.optimize(
            cluster_uplift, cluster_size, sigma, costs, budget_constraint
        )  # x.shape = (n_cluster, n_treatment)

        # sampling each cluster
        assign_df = pd.DataFrame(labels, columns=["cluster"])
        assign_df["assign"] = 0
        for cluster in range(self.n_cluster):
            for coupon in range(self.n_treatment):
                sample_df = assign_df.query("cluster == @cluster and assign == 0")
                n_assign = min(
                    int(x[cluster, coupon]), cluster_size[cluster], len(sample_df)
                )
                if len(sample_df) > 0:
                    sample_index = sample_df.sample(n=n_assign, random_state=seed).index
                    assign_df.loc[sample_index, "assign"] = coupon + 1

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

    def optimize(self, cluster_uplift, cluster_size, sigma, costs, budget_constraint):
        x = cp.Variable(shape=cluster_uplift.size, nonneg=True)
        y = cp.Variable(shape=cluster_uplift.size, nonneg=True)
        zp = cp.Variable(shape=sigma.size + 1, nonneg=True)

        # c^T x + z0 * gamma + sum_{p0j}
        objective = cp.Minimize(
            -(x @ cluster_uplift.T)
            + zp
            @ np.hstack(
                [int(self.postprocess_config.params["gamma"]), np.ones_like(sigma)]
            ).T
        )

        constraints = []
        # budget constraint
        constraints.append(x @ costs.T <= budget_constraint)

        # cluster size constraint
        for i in range(self.n_cluster):
            constraints.append(
                cp.sum(x[self.n_treatment * i : self.n_treatment * (i + 1)])
                <= cluster_size[i]
            )

        # z0 + p0j <= d_j * y_j
        constraints.append(zp[0] + zp[1:] >= cp.multiply(y, sigma))

        # -y <= x <= y
        constraints.append(-y <= x)
        constraints.append(x <= y)

        # z_0 >= 0
        constraints.append(zp[0] >= 0)

        # solve
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        x = x.value.reshape(self.n_cluster, self.n_treatment)

        self.logger.info(f"Optimal value: {prob.value}")
        self.logger.info(f" Sufficiency ratio: {(x*self.cost).sum()/budget_constraint:3f}")

        return x
