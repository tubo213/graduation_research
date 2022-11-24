import numpy as np


class GreedyPostprocess:
    def __init__(self, postprocess_config):
        self.postprocess_config = postprocess_config
        self.method = postprocess_config.params["method"]

    def postprocess(self, uplift_mat, budget_constraint):
        customer_list = list(range(uplift_mat.shape[0]))
        coupon_list = list(range(uplift_mat.shape[1]))
        # uplift/cost の降順にソート
        if self.method == "cpa_greedy":
            customer_coupon_list = [
                [
                    float(uplift_mat[customer, coupon])
                    / self.postprocess_config.variant_no_to_cost[coupon + 1],
                    customer,
                    coupon,
                ]
                for customer in customer_list
                for coupon in coupon_list
            ]
        # uolift の降順にソート
        elif self.method == "gmv_greedy":
            customer_coupon_list = [
                [
                    float(uplift_mat[customer, coupon]),
                    customer,
                    coupon,
                ]
                for customer in customer_list
                for coupon in coupon_list
            ]
        # 評価指標の降順にソート
        customer_coupon_list.sort(reverse=True)
        # すでに割り当てられたcustomerの集合
        assigned_customer = set()
        # customerに割り当てるcouponのdict
        customer2coupon = {}
        for priority, customer, coupon in customer_coupon_list:
            # 割り当てられていたらスキップ
            if customer in assigned_customer:
                continue
            # 予算を超えていたらスキップ
            if (
                self.postprocess_config.variant_no_to_cost[coupon + 1]
                > budget_constraint
            ):
                continue
            # priorityの推定値が負ならスキップ
            if priority <= 0:
                continue
            # 割り当てれたら予算を減らす
            budget_constraint -= self.postprocess_config.variant_no_to_cost[coupon + 1]
            customer2coupon[customer] = coupon + 1
            assigned_customer.add(customer)

        # customerの長さのlistにcoupon_noを入れる
        assginment = [
            customer2coupon[customer] if customer in customer2coupon else 0
            for customer in customer_list
        ]
        assert len(assginment) == uplift_mat.shape[0]
        return np.array(assginment)
