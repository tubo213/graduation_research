import multiprocessing

import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src.utils import save_as_pickle


class ClusterPostprocess:
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

        problem = pulp.LpProblem("CouponAllocation", pulp.LpMaximize)

        # 決定変数(0 <= x)
        x = pulp.LpVariable.dicts("x", customer_coupon_list, lowBound=0)

        # 配布数 <= クラスタサイズ
        for cluster in cluster_list:
            problem += (
                pulp.lpSum([x[cluster, coupon] for coupon in coupon_list])
                <= cluster_sizes[cluster]
            )

        # 予算制約
        problem += (
            pulp.lpSum(
                [
                    x[cluster, coupon]
                    * self.postprocess_config.variant_no_to_cost[coupon + 1]
                    for cluster in cluster_list
                    for coupon in coupon_list
                ]
            )
            <= budget_constraint
        )

        # 目的関数: upliftの総和を最大化
        problem += pulp.lpSum(
            [
                x[cluster, coupon] * uplift_mat[cluster, coupon]
                for cluster in cluster_list
                for coupon in coupon_list
            ]
        )

        # 求解
        _ = problem.solve(
            pulp.PULP_CBC_CMD(msg=False, threads=multiprocessing.cpu_count())
        )

        # (cluster_size, n_coupon)
        assginment = [
            [int(x[cluster, coupon].value()) for coupon in coupon_list]
            for cluster in cluster_list
        ]
        assert len(assginment) == len(cluster_list)

        # sampling each cluster
        cluster_df["assignment"] = 0
        for cluster in cluster_list:
            for coupon in coupon_list:
                n_assign = assginment[cluster][coupon]
                sample_index = (
                    cluster_df.query("cluster == @cluster and assignment == 0")
                    .sample(n_assign)
                    .index
                )
                cluster_df.loc[sample_index, "assignment"] = coupon + 1

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
