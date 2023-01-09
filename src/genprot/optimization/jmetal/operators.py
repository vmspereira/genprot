"""JMetalpy operators
"""

from optimization.jmetal.problem import protSolution
from utils.constants import EAConstants
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.solution import Solution
from typing import List
import random
import copy


class OnePointCrossover(Crossover[protSolution,protSolution]):
    """One point Crossover

    :param probability: (float) The probability of crossover.    
    """
    def __init__(self, probability: float = 0.1):
        super(OnePointCrossover, self).__init__(probability=probability)
        

    def execute(self, parents: List[protSolution]) -> List[protSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        mom = copy.copy(offspring[0].variables)
        dad = copy.copy(offspring[1].variables)    
        if random.random() <= self.probability:
            size = len(mom)
            cut_point = random.randint(1,size-2)
            bro = mom[:cut_point]+dad[cut_point:]
            sis = dad[:cut_point]+mom[cut_point:]
            offspring[0].variables=bro
            offspring[0].number_of_variables = len(bro)
            offspring[1].variables=sis
            offspring[1].number_of_variables = len(sis)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'One Point Crossover'


class TwoPointCrossover(Crossover[protSolution,protSolution]):
    """Two Point Crossover
    :param probability: (float) The probability of crossover.
    
    """
    def __init__(self, probability: float = 0.1):
        super(TwoPointCrossover, self).__init__(probability=probability)
        

    def execute(self, parents: List[protSolution]) -> List[protSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        mom = copy.copy(offspring[0].variables)
        dad = copy.copy(offspring[1].variables)    
        if random.random() <= self.probability:
            num_cuts = min(len(mom)-1,2)
            cut_points = random.sample(range(1, len(mom)), num_cuts)
            cut_points.sort()
            bro = copy.copy(dad)
            sis = copy.copy(mom)
            normal = True
            for i, (m, d) in enumerate(zip(mom, dad)):
                if i in cut_points:
                    normal = not normal
                if not normal:
                    bro[i] = m
                    sis[i] = d
                    normal = not normal
            offspring[0].variables=bro
            offspring[0].number_of_variables = len(bro)
            offspring[1].variables=sis
            offspring[1].number_of_variables = len(sis)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Two Point Crossover'



class MutationContainer(Mutation[Solution]):
    """A container for the mutation operators.
    
    :param probability: (float) The probability of applying a mutation.
    :param mutators: (list) The list of mutators.
    
    """
    
    def __init__(self,probability:float = 0.5, mutators = []):
        super(MutationContainer, self).__init__(probability=probability)
        self.mutators = mutators


    def execute(self, solution: Solution) -> Solution:
        # randomly select a mutator and apply it 
        if random.random() <= self.probability:
            idx = random.randint(0,len(self.mutators)-1)
            mutator = self.mutators[idx]
            return mutator.execute(solution)
        else:
            return solution


    def get_name(self):
        return 'Mutation container'




class GaussianMutation(Mutation[protSolution]):
    """
     A Gaussian mutator
    """
    def __init__(self, probability: float = 0.1, 
                 gaussian_mutation_rate: float =0.1,
                 gaussian_mean: float = 0.0,
                 gaussian_std: float =1.0):
        super(GaussianMutation, self).__init__(probability = probability)
        self.gaussian_mutation_rate = gaussian_mutation_rate
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        

    def execute(self, solution: Solution) -> Solution:
        if random.random() <= self.probability:
            mutant = copy.copy(solution.variables)
            for i, m in enumerate(mutant):
                if random.random() < self.gaussian_mutation_rate:
                    v = m + random.gauss(self.gaussian_mean, self.gaussian_std)
                    counter = 0
                    while v < solution.lower_bound[i] or v > solution.upper_bound[i]: 
                        if counter>100:
                            v = max(min(v, solution.upper_bound[i]), solution.lower_bound[i])
                            break
                        v = m + random.gauss(self.gaussian_mean, self.gaussian_std)
                        counter += 1
                    solution.variables[i]=v
        return solution



    def get_name(self):
        return 'Gaussian Mutator'




class SingleMutation(Mutation[protSolution]):
    """
    Mutates a single element
    """
    def __init__(self, probability: float = 0.1):
        super(SingleMutation, self).__init__(probability = probability)
        

    def execute(self, solution: Solution) -> Solution:
        if random.random() <= self.probability:
            index = random.randint(0, solution.number_of_variables - 1)
            solution.variables[index] = solution.lower_bound[index] + \
                                        (solution.upper_bound[index] - solution.lower_bound[index]) * random.random()
        return solution



    def get_name(self):
        return 'Single Mutation'



