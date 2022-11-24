import multiprocessing

import numpy as np
import pulp


class DiscretePostprocess:
    def __init__(self, postprocess_config):
        self.postprocess_config = postprocess_config

    def postprocess(self, uplift_mat, budget_constraint, seed):
        customer_list = list(range(uplift_mat.shape[0]))
        coupon_list = list(range(uplift_mat.shape[1]))
        customer_coupon_list = [
            (customer, coupon) for customer in customer_list for coupon in coupon_list
        ]

        problem = pulp.LpProblem("CouponAllocation", pulp.LpMaximize)

        # 決定変数(x in {0, 1})
        x = pulp.LpVariable.dicts("x", customer_coupon_list, cat="Binary")

        # 顧客一人に一つ以下のクーポン
        for customer in customer_list:
            problem += pulp.lpSum([x[customer, coupon] for coupon in coupon_list]) <= 1

        # 予算制約
        problem += (
            pulp.lpSum(
                [
                    x[customer, coupon]
                    * self.postprocess_config.variant_no_to_cost[coupon + 1]
                    for customer in customer_list
                    for coupon in coupon_list
                ]
            )
            <= budget_constraint
        )

        # 目的関数: upliftの総和を最大化
        problem += pulp.lpSum(
            [
                x[customer, coupon] * uplift_mat[customer, coupon]
                for customer in customer_list
                for coupon in coupon_list
            ]
        )

        # 求解
        _ = problem.solve(
            pulp.PULP_CBC_CMD(msg=False, threads=multiprocessing.cpu_count())
        )

        assginment = [
            max(
                [
                    coupon + 1 if x[customer, coupon].value() == 1 else 0
                    for coupon in coupon_list
                ]
            )
            for customer in customer_list
        ]
        assert len(assginment) == uplift_mat.shape[0]

        return np.array(assginment)
