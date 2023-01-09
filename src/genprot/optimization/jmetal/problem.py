""" JMetal Problems 
"""
from jmetal.core.solution import Solution, IntegerSolution
from jmetal.core.problem import Problem
from optimization.ea import SolutionInterface, dominance_test
import random
import warnings
from typing import List


class protSolution(Solution[float], SolutionInterface):
    """ Class representing a KO solution """

    def __init__(self, lower_bound: float, upper_bound: float, number_of_variables: int, number_of_objectives: int):
        super(protSolution, self).__init__(number_of_variables,
                                         number_of_objectives)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables.sort() == solution.variables.sort()
        return False

    # JMetal consideres all problems as minimization
    # Based on pareto dominance

    def __gt__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) == 1
        return False

    def __lt__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) == -1
        return False

    def __ge__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) != -1
        return False

    def __le__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=False) != 1
        return False

    def __copy__(self):
        new_solution = protSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()

        return new_solution

    def get_representation(self):
        """
        Returns a list representation of the candidate 
        """
        return self.variables

    def get_fitness(self):
        """
        Returns the candidate fitness list
        """
        return self.objectives

    def __str__(self):
        return " ".join((self.variables))



class JMetalProblem(Problem[protSolution]):

    def __init__(self, problem,batched=True):
        self.problem = problem
        self.number_of_objectives = len(self.problem.fevaluation)
        self.obj_directions = []
        self.obj_labels = []
        self.batched = batched
        for f in self.problem.fevaluation:
            self.obj_labels.append(str(f))
            if f.maximize:
                self.obj_directions.append(self.MAXIMIZE)
            else:
                self.obj_directions.append(self.MINIMIZE)

    def create_solution(self) -> protSolution:
        solution = self.problem.generator(random, None)
        new_solution = protSolution(
            self.problem.bounder.lower_bound,
            self.problem.bounder.upper_bound,
            len(solution),
            self.problem.number_of_objectives)
        new_solution.variables = list(solution)
        return new_solution

    def get_constraints(self, solution):
        return self.problem.decode(solution.variables)

    def evaluate_single(self, solution: protSolution) -> protSolution:
        candidate = solution.variables
        p = self.problem.evaluate_solution(candidate,self.batched)
        for i in range(len(p)):
            # JMetalPy only deals with minimization problems
            if self.obj_directions[i] == self.MAXIMIZE:
                solution.objectives[i] = -1 * p[i]
            else:
                solution.objectives[i] = p[i]
        return solution

    def evaluate_batch(self, solution_list: List[protSolution]) -> protSolution:
        
        listLatent = [solut.variables for solut in solution_list]
        listScores = self.problem.evaluate_solution( listLatent, self.batched)

        for i, solution in enumerate(solution_list):
            for j in range(len(listScores[i])):
                # JMetalPy only deals with minimization problems
                if self.obj_directions[j] == self.MAXIMIZE:
                    solution.objectives[j] = -1 * listScores[i][j]
                else:
                    solution.objectives[j] = listScores[i][j]

        return solution_list

    def evaluate(self, solution):
        if self.batched: return self.evaluate_batch(solution)
        return self.evaluate_single(solution)

    def get_name(self) -> str:
        return self.problem.get_name()

