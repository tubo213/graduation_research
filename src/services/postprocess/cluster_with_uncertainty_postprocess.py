import cvxpy as cvx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src.utils import save_as_pickle


class ClusterWithUncertaintyPostprocess:
    def __init__(self, base_config, postprocess_config):
        self.base_config = base_config
        self.postprocess_config = postprocess_config

    def postprocess(self, uplift_mat, budget_constraint: int, seed: int):
        # clustering
        # # normalize uplift_mat
        clustering_model = self.get_clustering()
        scaler = StandardScaler()
        normalized_uplift_mat = scaler.fit_transform(uplift_mat)
        clustering_model.fit(normalized_uplift_mat)
        save_clustering_model_path = (
            self.base_config.dir_config.output_model_dir
            / f"clustering_model_{seed}_{budget_constraint}.pkl"
        )
        save_scaler_path = (
            self.base_config.dir_config.output_model_dir
            / f"scaler_{seed}_{budget_constraint}.pkl"
        )
        save_as_pickle(clustering_model, save_clustering_model_path)
        save_as_pickle(scaler, save_scaler_path)

        # # get cluster labels
        cluster_df = pd.DataFrame(
            clustering_model.predict(normalized_uplift_mat), columns=["cluster"]
        )
        np.save(
            self.base_config.dir_config.output_optimize_dir
            / f"cluster_{seed}_{budget_constraint}.npy",
            cluster_df.to_numpy(),
        )

        cluster_sizes = cluster_df["cluster"].value_counts().sort_index().to_numpy()
        uplift_mat = (
            pd.DataFrame(uplift_mat).groupby(cluster_df["cluster"]).mean().to_numpy()
        )

        # solve optimization problem
        cluster_list = list(range(cluster_sizes.shape[0]))
        coupon_list = list(range(uplift_mat.shape[1]))
        customer_coupon_list = [
            (cluster, coupon) for cluster in cluster_list for coupon in coupon_list
        ]

        customer_coupon_list

        return cluster_df["assignment"].to_numpy()

    def get_clustering(self):
        if self.postprocess_config.params["clustering"] == "kmeans":
            return KMeans(
                n_clusters=self.postprocess_config.params["n_cluster"],
                random_state=self.base_config.seed,
            )
        elif self.postprocess_config.params["clustering"] == "gmm":
            return GaussianMixture(
                n_components=self.postprocess_config.params["n_cluster"],
                random_state=self.base_config.seed,
            )
