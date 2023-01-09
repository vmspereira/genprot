from jmetal.algorithm.singleobjective import SimulatedAnnealing
from jmetal.algorithm.multiobjective import NSGAII, SPEA2, GDE3
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import BinaryTournamentSelection
from utils.process import MultiProcessorEvaluator
from optimization.ea import AbstractEA
from .problem import JMetalProblem
from .observers import PrintObjectivesStatObserver, VisualizerObserver
from .operators import * 
from .generators import presetGenerator
from .evaluators import BatchEvaluator
from .terminators import StoppingByEvaluationsAndQuality
from .algorithms import mGeneticAlgorithm as GeneticAlgorithm
from utils.constants import EAConstants
from utils.process import cpu_count


# SOEA alternatives
soea_map ={
    'GA': GeneticAlgorithm,
    'SA': SimulatedAnnealing
}
# MOEA alternatives
moea_map = {
    'NSGAII': NSGAII,
    'SPEA2': SPEA2,
    'NSGAIII': NSGAIII,
    'GDE3':GDE3
}


class EA(AbstractEA):
    """
    EA running helper for JMetal.

    
    :param problem: The optimization problem.
    :param initial_population: (list) The EA initial population.
    :param max_generations: (int) The number of iterations of the EA (stopping criteria). 
    """

    def __init__(self, problem, initial_population=[], max_generations=EAConstants.MAX_GENERATIONS, mp=True,
                       visualizer=False, algorithm = None, batched=True, configs=None):

        super(EA, self).__init__(problem, initial_population=initial_population,
                                 max_generations=max_generations, mp=mp, visualizer=visualizer)
        self.algorithm_name = algorithm
        self.ea_problem = JMetalProblem(self.problem,batched=batched)
        self.crossover = OnePointCrossover(0.6)
        mutators = []
        mutators.append(SingleMutation(1))
        mutators.append(GaussianMutation(1))
        self.mutation = MutationContainer(0.75, mutators=mutators)
        self.configs = configs
        self.population_evaluator = BatchEvaluator()
        self.population_size = 10 
        self.termination_criterion = StoppingByEvaluations(max_evaluations=self.max_generations * self.population_size)

    def _run_so(self):
        """ Runs a single objective EA optimization ()
        """
        
        
        if self.algorithm_name == 'SA':
            print("Running SA")
            self.mutation.probability = 1.0
            algorithm = SimulatedAnnealing(
                problem=self.ea_problem,
                mutation=self.mutation.probability, 
                termination_criterion=self.termination_criterion
            )
        else:
            print("Running GA")
            algorithm = GeneticAlgorithm(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
                selection=BinaryTournamentSelection(),
                termination_criterion=self.termination_criterion,
                population_evaluator=self.population_evaluator
            )

        algorithm.observable.register(observer=PrintObjectivesStatObserver())
        algorithm.run()

        result = algorithm.solutions
        return result

    def _run_mo(self):
        """ Runs a multi objective EA optimization
        """
        max_evaluations = self.max_generations * self.population_size
        ncpu = cpu_count()
        if self.algorithm_name in moea_map.keys():
            f = moea_map[self.algorithm_name]
        else:
            if self.ea_problem.number_of_objectives > 2:
                self.algorithm_name== 'NSGAIII'
            else:
                f = moea_map['SPEA2']

        print(f"Running {self.algorithm_name}")
        if self.algorithm_name== 'NSGAIII':
            algorithm = NSGAIII(
                problem=self.ea_problem,
                population_size=self.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
                termination_criterion=self.termination_criterion,
                reference_directions=UniformReferenceDirectionFactory(self.ea_problem.number_of_objectives, n_points=self.population_size-1),
                population_evaluator=self.population_evaluator
            )
        elif self.algorithm_name== "GDE3":
            algorithm = f(
                problem=self.ea_problem,
                population_size=self.population_size,
                cr=0.5,
                f=0.5,
                termination_criterion=self.termination_criterion,
                population_evaluator=self.population_evaluator
            )
        else:
            algorithm = f(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
                termination_criterion=self.termination_criterion,
                population_evaluator=self.population_evaluator
            )

        if self.visualizer:
            algorithm.observable.register(observer=VisualizerObserver())
        algorithm.observable.register(observer=PrintObjectivesStatObserver())

        algorithm.run()
        result = algorithm.solutions
        return result

    def _convertPopulation(self, population):
        p = []
        from optimization.ea import Solution as ProblemSolution
        for i in range(len(population)):
            # Corrects fitness values for maximization problems
            # TODO: verify each objective individualy
            if self.problem.is_maximization:
                obj = [abs(x) for x in population[i].objectives]
            else:
                obj = [x for x in population[i].objectives]
            val = population[i].variables[:]
            #const = self.problem.decode(val)
            solution = ProblemSolution(val, obj)
            p.append(solution)
        return p
