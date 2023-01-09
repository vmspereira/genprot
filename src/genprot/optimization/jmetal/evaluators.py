from jmetal.util.evaluator import Evaluator
from jmetal.core.problem import Problem
from typing import TypeVar, List

S = TypeVar('S')


class BatchEvaluator(Evaluator):

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        Evaluator.evaluate_solution(solution_list, problem)

        return solution_list