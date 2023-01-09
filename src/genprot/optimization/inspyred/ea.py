from utils.process import MultiProcessorEvaluator, cpu_count
from utils.constants import EAConstants
from ..ea import AbstractEA, Solution
from .problem import InspyredProblem
from . import operators
from . import observers
from random import Random
from time import time
import inspyred



SOEA={
    'GA':inspyred.ec.EvolutionaryComputation,
    'SA':inspyred.ec.SA
}

class EA(AbstractEA):
    """
    EA running helper

    :param problem: the optimization problem.
    :param initial_population: (list) the EA initial population.
    :param max_generations: (int) the number of iterations of the EA (stopping criteria).
    """

    def __init__(self, problem, initial_population=[], max_generations=EAConstants.MAX_GENERATIONS, mp=True,
                       visualizer=False, algorithm=None, batched=True, configs=None):

        super(EA, self).__init__(problem, initial_population=initial_population,
                                 max_generations=max_generations, mp=mp, visualizer=visualizer)

        self.algorithm_name = algorithm
        self.ea_problem = InspyredProblem(self.problem, batched=batched)
        self.configs = configs
        self.all_mols = {}
        self.variators = [operators.one_point_crossover,
                   operators.two_point_crossover,
                   operators.real_arithmetical_crossover,
                   operators.gaussian_mutation
                  ]

        

                ## needs to be defined elsewhere
        self.args = { 
                'num_selected' : 100,
                'max_generations' :self.max_generations,
                # operators probabilities
                'one_point_crossover_rate' : 0.1,
                'two_point_crossover_rate' : 0.1,
                'real_arithmetical_crossover_rate' : 0.4,
                'gaussian_mutation_rate' : 0.4,
                # operators confs
                'num_mix_points' : 5,
                'gaussian_gene_mutation' : 0.1
                }
        
        if self.problem.number_of_objectives == 1:
            self.args['tournament_size'] = 7


    def _run_so(self):
        """ Runs a single objective EA optimization """

        prng = Random()
        prng.seed(time())

        if self.mp:
            raise NotImplementedError
        else:
            self.evaluator = self.ea_problem.evaluator

        if self.algorithm_name == 'SA':
            ea = inspyred.ec.SA(prng)
            print("Running SA")
        else:
            ea = inspyred.ec.EvolutionaryComputation(prng)
            print("Running GA")
        ea.selector = inspyred.ec.selectors.tournament_selection

        ea.variator = self.variators
        ea.observer = observers.Observers(self.all_mols,self.configs)
        ea.replacer = inspyred.ec.replacers.truncation_replacement
        ea.terminator = inspyred.ec.terminators.generation_termination

        final_pop = ea.evolve(generator=self.problem.generator,
                              evaluator=self.evaluator,
                              pop_size=100,
                              seeds=self.initial_population,
                              maximize=self.problem.is_maximization,
                              bounder=self.problem.bounder,
                              **self.args
                              )
        self.final_population = final_pop
        return final_pop

    def _run_mo(self):
        """ Runs a multi objective EA (NSGAII) optimization
        """
        prng = Random()
        prng.seed(time())

        if self.mp:
            nmp = cpu_count()
            try:
                from mewpy.utils.process import RayEvaluator
                mp_evaluator = RayEvaluator(self.ea_problem, nmp)
            except ImportError:
                mp_evaluator = MultiProcessorEvaluator(
                    self.ea_problem.evaluate, nmp)
            self.evaluator = mp_evaluator.evaluate
        else:
            self.evaluator = self.ea_problem.evaluator

        ea = inspyred.ec.emo.NSGA2(prng)
        print("Running NSGAII")
        ea.variator = self.variators
        ea.terminator = inspyred.ec.terminators.generation_termination
        if self.visualizer:
            axis_labels = [f.short_str() for f in self.problem.fevaluation]
            observer = observers.VisualizerObserver(self.all_mols,self.configs,axis_labels=axis_labels)
            ea.observer = observer.update
        else:
            obs = observers.Observers(self.all_mols,self.configs)
            ea.observer = obs.results_observer

        final_pop = ea.evolve(generator=self.problem.generator,
                              evaluator=self.evaluator,
                              pop_size=100,
                              seeds=self.initial_population,
                              maximize=self.problem.is_maximization,
                              bounder=self.problem.bounder,
                              **self.args
                              )

        self.final_population = final_pop
        return final_pop

    def _convertPopulation(self, population):
        p = []
        for i in range(len(population)):
            if self.problem.number_of_objectives == 1:
                obj = [population[i].fitness]
            else:
                obj = population[i].fitness
            val = population[i].candidate
            #values = self.problem.translate(val)
            #const = self.problem.decode(val)
            solution = Solution(val, obj)
            p.append(solution)
        return p
