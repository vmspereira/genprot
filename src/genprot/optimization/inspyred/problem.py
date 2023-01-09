from inspyred.ec.emo import Pareto




class InspyredProblem:
    """Inspyred EA builder helper.

        :param problem: the optimization problem.
    """

    def __init__(self, problem, batched=True):
        self.problem = problem
        self.batched = batched

    def evaluate(self, solution):
        """Evaluates a single solution

            :param solution: The individual to be evaluated.
            :returns: A list with a fitness value or a Pareto object.

        """
        p = self.problem.evaluate_solution(solution)
        # single objective
        if self.problem.number_of_objectives == 1:
            return p[0]
        # multi objective
        else:
            return Pareto(p)

    def evaluator(self, candidates, args):
        """
        Evaluator
        Note: shoudn't be dependent on args to ease multiprocessing

        :param candidates: A list of candidate solutions.
        :returns: A list of Pareto fitness values or a list of fitness values.

        """

        listScores = self.problem.evaluate_solution( candidates, batched=self.batched)
        fitness = []

        for i in range(len(candidates)):

            if self.problem.number_of_objectives == 1:
                if self.batched: fitness.append( listScores[i][0] )
                else: fitness.append( listScores[i] )
            else:
                fitness.append( Pareto(listScores[i]) )

        return fitness