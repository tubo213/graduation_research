import multiprocessing
from dataclasses import dataclass

import numpy as np
import pulp
from src.services.base.base_service import BaseService
from src.services.config.config_service import ConfigService
from src.services.data.data_service import DataService


@dataclass
class PostprocessService(BaseService):
    def __init__(self, config_service: ConfigService, data_service: DataService):
        super().__init__()
        self.config_service = config_service
        self.postprocess_config = self.config_service.config.postprocess_config
        self.data_service = data_service

    def postprocess(self, uplift_mat, budget_constraint, timelimit=100):
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
            pulp.PULP_CBC_CMD(
                msg=False, threads=multiprocessing.cpu_count(), timeLimit=timelimit
            )
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
