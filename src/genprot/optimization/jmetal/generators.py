from jmetal.util.generator import Generator
from .problem import protSolution 
from ..problem import RealBounder

class presetGenerator(Generator):
    def __init__(self, initial_population):
        super(presetGenerator, self).__init__()
        self.initial_population = initial_population
        self.curr = 0
        #self.bounder = RealBounder(len(self.initial_population[0]), -10.0 , 10.0)

    def new(self, problem):
        if self.curr == len(self.initial_population): self.curr = 0
        individual = self.initial_population[self.curr]
        new_solution = protSolution(
            problem.problem.bounder.lower_bound,
            problem.problem.bounder.upper_bound,
            len(individual),
            problem.number_of_objectives)
        new_solution.variables = list(individual)
        self.curr += 1
        return new_solution