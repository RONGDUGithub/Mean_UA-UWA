from math import log
from typing import Dict, List
from evaluate_module.evaluate_module_abc import EvaluateModuleABC


class EvaluateModule(EvaluateModuleABC):
    def __init__(self, evaluate_type: str = "F1"):
        """_summary_

        Args:
            top_k (int): for `top-k` heavy hitters
            evaluate_type (str, optional): _description_. Defaults to "NDCG".
        """
        self.evaluate_type = evaluate_type

    def F1(self, truth_top_k: List, estimate_top_k: List):
        """_summary_

        Args:
            truth_top_k (List): The real top k heavy hitters
            estimate_top_k (List): Estimated top k heavy hitters

        Returns:
            _type_: f1 score
        """
        if truth_top_k is None:
            return -1
        if len(estimate_top_k) == 0:
            return 0
        top_k = len(truth_top_k)
        truth_top_k = truth_top_k[:top_k]
        hit = 0
        for hitter in estimate_top_k:
            if hitter in truth_top_k: hit += 1
        precision = hit / len(estimate_top_k)
        recall = hit / len(truth_top_k)
        if precision == 0 and recall == 0: return 0
        return 2 * precision * recall / (precision + recall)

    def recall(self, truth_top_k, estimate_top_k):
        """_summary_

        Args:
            truth_top_k (list): The real top k heavy hitters
            estimate_top_k (list): Estimated top k heavy hitters
            k (int): top-k heavy hitters
        Returns:
            float: Recall
        """
        if truth_top_k is None:
            return -1
        top_k = len(truth_top_k)
        print(f"Find {len(estimate_top_k)} top-k heavy hitters")
        truth_top_k = truth_top_k[:top_k]
        hit = 0
        for hitter in estimate_top_k:
            if hitter in truth_top_k: hit += 1
        return hit / len(truth_top_k)

    def __DCG(self, estimate_top_k: List, rel: Dict) -> float:
        """_summary_

        Args:
            estimate_top_k (List):Estimated top k heavy hitters
            rel (Dict): relevance of each element in estimate_top_k

        Returns:
            float: Discount Cumulative Gain
            """
        DCG = 0
        for indx, item in enumerate(estimate_top_k):
            rel_ = 2 ** rel.get(item, 0) - 1
            DCG += rel_ / log(indx + 2)
        return DCG

    def NDCG(self, truth_top_k, estimate_top_k):
        """_summary_

        Args:
            truth_top_k (list): The real top k heavy hitters
            estimate_top_k (list): Estimated top k heavy hitters
            k (int): top-k heavy hitters
        Returns:
            float: Normalized Discount Cumulative Gain
        """
        top_k = self.top_k
        print(f"Find {len(estimate_top_k)} top-k heavy hitters")
        truth_top_k = truth_top_k[:top_k]
        rel = {}
        for number in truth_top_k:
            rel[number] = log(top_k)
            top_k -= 1

        return self.__DCG(estimate_top_k, rel) / self.__DCG(truth_top_k, rel)

    def evaluate(self, truth_top_k, estimate_top_k):
        if self.evaluate_type == "NDCG":
            return self.NDCG(truth_top_k, estimate_top_k)
        elif self.evaluate_type == "F1":
            return self.F1(truth_top_k, estimate_top_k)
        elif self.evaluate_type == "recall":
            return self.recall(truth_top_k, estimate_top_k)