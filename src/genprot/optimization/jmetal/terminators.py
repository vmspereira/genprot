from jmetal.util.termination_criterion import TerminationCriterion
from optimization.problem import proteinProblem
import numpy as np
import tensorflow as tf 
from utils.decoding import _decode_ar


class StoppingByEvaluationsAndQuality(TerminationCriterion):

    def __init__(self, problem: proteinProblem, max_evaluations: int, expected_value: float):
        super(StoppingByEvaluationsAndQuality, self).__init__()
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.expected_value = expected_value
        self.evaluations = 0
        self.value = 0.0
        self.checkPoints = [max_evaluations, max_evaluations*2, max_evaluations*5]

    def update(self, *args, **kwargs):
        self.evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if solutions and self.evaluations in self.checkPoints:
            if "Aggregated Sum" in self.problem.fevaluation[0].method_str():
                listLatent = [s.variables for s in solutions]
                c = tf.convert_to_tensor(np.asarray(listLatent)).float().cuda()
                listPROTS = self.problem.gen_model.generateMols(latents=c)
                listMols = [_decode_ar(prot) for prot in listPROTS]
                evals = self.problem.fevaluation[0].fevaluation[-1](listMols,batched=True)
    
            else:
                evals = [solut.objectives[-1]*-1 for solut in solutions]
            
            evals = [ev/100 for ev in evals]
            
            self.value = np.max(evals)

    @property
    def is_met(self):
        met = (self.value > self.expected_value) and (self.evaluations >= self.max_evaluations)
        met = met or (self.evaluations >= self.max_evaluations*10) 

        return met
