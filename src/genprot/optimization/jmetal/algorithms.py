from typing import TypeVar
from jmetal.algorithm.singleobjective import GeneticAlgorithm

R = TypeVar('R')


class mGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, **kwarg):
        super(mGeneticAlgorithm, self).__init__(**kwarg)
        
    def get_result(self) -> R:
        return self.solutions