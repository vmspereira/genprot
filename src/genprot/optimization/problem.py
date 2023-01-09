import sys
sys.path.append('/home/mmartins/GenProtEA')
from loadModels import loadVAE
import numbers
import collections
import numpy as np


'''
Reconstruction of individuals from the population into protein sequences,
scoring them based on the classes defined in the evaluation module
'''



class RealBounder(object):

    def __init__(self, solution_size,lower_bound=None, upper_bound=None):
        
        self.solution_size = solution_size

        if lower_bound is not None and isinstance(lower_bound, numbers.Number):
            self.lower_bound = [lower_bound] * solution_size
        elif lower_bound and isinstance(lower_bound, list) and len(lower_bound)==self.solution_size:
            self.lower_bound = lower_bound
        else: 
            self.lower_bound = None


        if upper_bound  is not None and isinstance(upper_bound, numbers.Number):
            self.upper_bound = [upper_bound] * solution_size
        elif upper_bound and isinstance(upper_bound, list) and len(upper_bound)==self.solution_size:
            self.upper_bound = upper_bound
        else:
            self.upper_bound = None
            
            


    def __call__(self, candidate, args):
        
        """

        """
        if self.lower_bound is None or self.upper_bound is None:
            return candidate
        else:
            if not isinstance(self.lower_bound, collections.Iterable):
                self.lower_bound = [self.lower_bound] * len(candidate)
            if not isinstance(self.upper_bound, collections.Iterable):
                self.upper_bound = [self.upper_bound] * len(candidate)
            bounded_candidate = candidate
            for i, (c, lo, hi) in enumerate(zip(candidate, self.lower_bound, self.upper_bound)):
                bounded_candidate[i] = max(min(c, hi), lo)
            return bounded_candidate




class Problem(object):

    def __init__(self, name, fevaluation, dimensions=100):
        if fevaluation is None:
            raise ValueError("At least one evaluation function needs to be provided")
        self.fevaluation = fevaluation
        self.name = name
        self.dimensions = dimensions
        self.number_of_objectives = len(self.fevaluation)
        self.bounder = None

    @property
    def is_maximization(self):
        return all([f.maximize for f in self.fevaluation])    
        
    def __str__(self):
        if self.number_of_objectives > 1:
            return '{0} ({1} dimensions, {2} objectives)'.format(self.__class__.__name__, self.dimensions, self.number_of_objectives)
        else:
            return '{0} ({1} dimensions)'.format(self.__class__.__name__, self.dimensions)
        
    def __repr__(self):
        return self.__class__.__name__
    
    def generator(self, random, args):
        """The generator function for the problem."""
        raise NotImplementedError
        
    def evaluate(self, candidate, args):
        """ Evaluates a single candidate"""
        raise NotImplementedError

    def evaluator(self, candidates, args):
        """
        Evaluator 
        returns a list of Pareto fitness values of a candidate list
        """
        fitness = []
        for candidate in candidates:
            p = self.evaluate(candidate,args)
            fitness.append(p)
        return fitness
    
    def _evaluate_solution_single(self, latentVect):
        """ Evaluates a single solution """
        raise NotImplementedError
    
    def _evaluate_solution_batch(self, listLatent):
        """ 
            Evaluates a batch of solutions 
            Returns: A list of tuples, one tuple per individual.
                     Each tuple contains the scores, calculated with self.fevaluation, for that individual.
        """
        raise NotImplementedError
    
    def evaluate_solution(self, candidate, batched=True):
        """ Evaluates a candidate """
        raise NotImplementedError


class proteinProblem(Problem):

    def __init__(self, fevaluation):
        Problem.__init__(self, "protGen", fevaluation)
        self.gen_model = loadVAE()
        self.dimensions =self.gen_model.latent_dim
        self.bounder = RealBounder(self.dimensions, 0 , 100.0)
        

    def instanciate(self):
        pass
        
    def generator(self, random, args):
        """
        Generates a random real solution vector
        with bounded values
        """
        solution = [] 
        for i in range(self.dimensions):
            lb = self.bounder.lower_bound[i]
            ub = self.bounder.upper_bound[i]
            if lb is not None and ub is not None : 
                solution.append(random.uniform(lb, ub))
            else: 
                solution.append(random.rafevaluationndom())
        return solution

    def _evaluate_solution_single(self, latentVect):
        """ Evaluates a single solution """

        def to_array(candidate):
            return np.expand_dims( np.asarray(candidate) , 0)

        p = []
        c = to_array(latentVect)
        decodProt = self.gen_model.decode(c)
        print(decodProt)
        
        for f in self.fevaluation: p.append(f(decodProt,batched=False))

        return p


    def _evaluate_solution_batch(self, listLatent):
        """ 
            Evaluates a batch of solutions 
            Returns: A list of tuples, one tuple per individual.
                     Each tuple contains the scores, calculated with self.fevaluation, for that individual.
        """

        c = np.asarray(listLatent)
        listPROTS = self.gen_model.decode(c)
        evals = []
        for f in self.fevaluation:
            if f(listPROTS,batched=True) is not None:
                evals.append( f(listPROTS,batched=True))
        
        return list(zip(*evals))
        

    def evaluate_solution(self, candidate, batched=True):
        """ Evaluates a candidate """

        if not self.gen_model: self.instanciate()

        if batched: return self._evaluate_solution_batch(candidate)
        
        return self._evaluate_solution_single(candidate)



    def copy(self):
        fs = [f.copy() for f in self.fevaluation]
        return proteinProblem(fs)



class proteinReporter(proteinProblem):

    def __init__(self, fevaluation):
        super(proteinReporter,self).__init__(fevaluation)

    def _evaluate_solution_single(self, latentVect):
        """ Evaluates a single solution """

        def to_array(candidate):
            return np.expand_dims( np.asarray(candidate) , 0)

        p = []
        c = to_array(latentVect)
        listPROTS = self.gen_model.decode(c)

        for f in self.fevaluation: p.append(f(listPROTS,batched=False))

        return p, listPROTS


    def _evaluate_solution_batch(self, listLatent):
        """
            Evaluates a batch of solutions
            Returns: A list of tuples, one tuple per individual.
                     Each tuple contains the scores, calculated with self.fevaluation, for that individual.
                     The decoded Proteins
        """

        c = np.asarray(listLatent)
        listPROTS = self.gen_model.decode(c)

        evals = []
        for f in self.fevaluation: evals.append( f(listPROTS,batched=True) )

        return list(zip(*evals)), listPROTS
