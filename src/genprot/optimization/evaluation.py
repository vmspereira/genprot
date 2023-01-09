from abc import ABCMeta, abstractmethod
import numpy as np
from functools import reduce
from hmmer.app import *
import math 
import re

hydroscale  = {'A':  0.620,'R': -2.530,'N': -0.780,'D': -0.900,
                'C':  0.290,'Q': -0.850,'E': -0.740,'G':  0.480,
                'H': -0.400,'Y':  0.260,'I':  1.380,'L':  1.060,
                'K': -1.500,'M':  0.640,'F':  1.190,'P':  0.120,
                'S': -0.180,'T': -0.050,'W':  0.810,'V':  1.080}
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
              'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
              
def remove_gaps(candidate):
    candidate = candidate.replace("-","")

    return candidate

def remove_X(candidate):
    for aa in candidate:
        if aa not in alphabet:
            candidate = candidate.replace(aa,"")
    return candidate

def Average(lst):
    return sum(lst) / len(lst)


def charge(candidate):

    pH=7
    netCharge=0.0
    pka_alpha_amino={'G':9.60,'A':9.69,'V':9.62,'L':9.60,'I':9.68,'M':9.21,'F':9.13,'W':9.39,'P':10.60,'S':9.15,
                     'T':9.10,'C':10.78,'Y':9.11,'N':8.84,'Q':9.13,'D':9.82,'E':9.67,'K':8.95,'R':9.04,'H':9.17}
    pka_alpha_carboxy={'G':2.34,'A':2.34,'V':2.32,'L':2.36,'I':2.36,'M':2.28,'F':1.83,'W':2.38,'P':1.99,'S':2.21,
                       'T':2.63,'C':1.71,'Y':2.2,'N':2.02,'Q':2.17,'D':2.09,'E':2.19,'K':2.18,'R':2.17,'H':1.82}
    pka_sidechain_positive={'K':10.79,'R':12.48,'H':6.04}
    pka_sidechain_negative={'D':3.86,'E':4.25,'C':8.33,'Y':10.07}
        
   
    # Calculate the net charge for the extreme groups (without modifications)
    candidate = remove_gaps(candidate)
    candidate = candidate.upper()
    amino=candidate[0]
    pos = 1
    carboxy=candidate[-pos]
    netCharge+= (math.pow(10,pka_alpha_amino[amino])/(math.pow(10,pka_alpha_amino[amino])+math.pow(10,pH)))
    netCharge-= math.pow(10,pH)/(math.pow(10,pka_alpha_carboxy[carboxy])+math.pow(10,pH))    
    
    # Calculate the net charge for the charged amino acid side chains

    for aa in candidate:
        if aa in pka_sidechain_positive:
            netCharge+=(math.pow(10,pka_sidechain_positive[aa])/(math.pow(10,pka_sidechain_positive[aa])+math.pow(10,pH)))
        if aa in pka_sidechain_negative:
            netCharge-=(math.pow(10,pH)/(math.pow(10,pka_sidechain_negative[aa])+math.pow(10,pH)))
    
    return netCharge

def solubility_rules(candidate):
    candidate = remove_gaps(candidate)
    candidate = candidate.upper()
    # Rule N1. Number of hydrophobic or charged residues
    hydro_residues=['V','I','L','M','F','W','C']
    charged_residues=['H','R','K','D','E']
    solubility_rules_failed = 0        
        
    count_hydro_charged=0
    for aa in candidate:
        if aa in hydro_residues or aa in charged_residues: count_hydro_charged+=1
        

    # This condition should change depending on the sequence length
    length_peptide = len(candidate)
    hydro_char_threshold=float(length_peptide)*0.45
    if count_hydro_charged > hydro_char_threshold:
        solubility_rules_failed+=1

    # Rule N2. Computed peptide charge
    charge_threshold=1
    try:
        
        if charge(candidate) > charge_threshold:
            solubility_rules_failed+=1
            
    except:
        print(candidate)
        return 1000000000

            
    # Rule N3. Glycine or Proline content in the sequence
    count_gly_pro=0
    for aa in candidate:
        if aa == "G" or aa=="P": 
            count_gly_pro+=1

    # Check threshold
    if count_gly_pro > 1:
        solubility_rules_failed+=1
        
    # Rule N4. First or last amino acid charged
    count_charge=0
    if candidate[0] in charged_residues:
        count_charge+=1
    if candidate[-1] in charged_residues:
        count_charge+=1
    # Check threshold
    if count_charge > 0:
        solubility_rules_failed+=1

        
    # Rule N5. Any amino acid represent more than 25% of the total sequence
    count_dict={"A":0,"R":0,"N":0,"D":0,"C":0,"Q":0,"E":0,"G":0,"H":0,"I":0,"L":0,"K":0,"M":0,"F":0,"P":0,"S":0,"T":0,"W":0,"Y":0,"V":0}
    for aa in candidate:
        if aa in count_dict.keys():
            count_dict[aa] +=1
    for i in count_dict.values():
        if i > 0.25*len(candidate):
            solubility_rules_failed +=1
    for aa in candidate:
        if aa =='-':
            solubility_rules_failed +=1
    return solubility_rules_failed

def synthesis_rules(candidate):
    
    candidate = remove_gaps(candidate)
    candidate = candidate.upper()
    #candidate = str(candidate)
    synthesis_rules_failed = 0

    forbidden_motifs = {'2-prolines':r'[P]{3,}','DG-DP':r'D[GP]'}
    for motif in forbidden_motifs:
        if re.search(forbidden_motifs[motif],candidate):
            synthesis_rules_failed+=1
            
    #check if sequence ends with a N or Q
    pos = 1
    try:
        terminal = candidate[-pos]
        if terminal == 'N' or terminal == 'Q':
            synthesis_rules_failed+=1
    except:
        print(candidate)
        return 1000000000
    
    # test if there are charged residues every 5 amino acids
    charged_residues=['H','R','K','D','E']
    counter_charged = 0
    for residue in candidate:
        counter_charged += 1
        if residue in charged_residues:
            counter_charged = 0
        if counter_charged >= 5:
            synthesis_rules_failed+=1
            
                
    # Check if there are oxidation-sensitive amino acids
    aa_oxidation=['M','C','W']
    for aa in candidate:
        if aa in aa_oxidation:
            synthesis_rules_failed+=1
            

    return synthesis_rules_failed




class EvaluationFunction:
    """
    This abstract class should be extended by all evaluation functions.

    """
    __metaclass__ = ABCMeta

    def __init__(self, maximize = True, worst_fitness = -1):
        self.worst_fitness = worst_fitness
        self.maximize = maximize


    @abstractmethod
    def _get_fitness_single(self, candidate):
        """
        Candidate :  Candidate beeing evaluated
        """
        return

    @abstractmethod
    def _get_fitness_batch(self, listMols):
        """
        listMols :  List of rdKit Mols beeing evaluated
        """
        return

    def get_fitness(self, candidate, batched):
        if batched:
            return self._get_fitness_batch(candidate)
        else:
            return self._get_fitness_single(candidate)

    @abstractmethod
    def method_str(self):
        return

    @abstractmethod
    def short_str(self):
        return ""

    def __str__(self):
        return self.method_str()

    
    def __call__(self, candidate, batched):
        return self.get_fitness(candidate, batched)
    
    
class AggregatedSum(EvaluationFunction):
    """
    Aggredated Sum Evaluation Function 

    Arguments:
        fevaluation (list): list of evaluation functions
        tradeoffs (list) : tradeoff values for each evaluation function. If None, all functions have the same weight

    """

    def __init__(self, fevaluation, tradeoffs=None, maximize = True):
        super(AggregatedSum,self).__init__(maximize= maximize, worst_fitness= 0.0)
        self.fevaluation = fevaluation
        if tradeoffs and len(tradeoffs) == len(fevaluation):
            self.tradeoffs = np.array(tradeoffs)
        else:
            self.tradeoffs = np.array([1/len(self.fevaluation)] * (len(self.fevaluation)))

    def _get_fitness_single(self, candidate):
        res = []
        for f in self.fevaluation:
            res.append(f._get_fitness_single(candidate))
        return np.dot(res, self.tradeoffs)
    
    def _get_fitness_batch(self, listProts):

        evals = []
        for f in self.fevaluation: evals.append( f._get_fitness_batch(listProts))
        evals = np.transpose( np.array(evals) )
        res = np.dot(evals, self.tradeoffs)
        return res


    def method_str(self):
        return "Aggregated Sum = " + reduce(lambda a, b: a+" "+b, [f.method_str() for f in self.fevaluation], "")

class Min_Rules_Solubility(EvaluationFunction):
    def __init__(self, maximize=False, worst_fitness=5):
        super(Min_Rules_Solubility,self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate):
        rules_failed = solubility_rules(candidate)#/len(candidate)
        return rules_failed

    def _get_fitness_batch(self, listProts):
        count_batch = []

        for candidate in listProts:
            rules_failed = solubility_rules(candidate)#/len(candidate)
            count_batch.append(rules_failed)

        return count_batch

    def __str__(self):
        return 'Solubility Rules Failed'

class Min_Rules_Synthesis(EvaluationFunction):
    def __init__(self, maximize=False, worst_fitness=100):
        super(Min_Rules_Synthesis,self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate):
        rules_failed = synthesis_rules(candidate)#/len(candidate)

        return rules_failed

    def _get_fitness_batch(self, listProts):
        count_batch = []

        for candidate in listProts:
            rules_failed = synthesis_rules(candidate)#/len(candidate)
            count_batch.append(rules_failed)

        return count_batch

    def __str__(self):
        return 'Synthesis Rules Failed'
        
        
        
class Min_reps(EvaluationFunction):
    def __init__(self, maximize=False, worst_fitness=10):
        super(Min_reps,self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate):
        reps_rules_failed = 0 
        count_dict={"A":0,"R":0,"N":0,"D":0,"C":0,"Q":0,"E":0,"G":0,"H":0,"I":0,"L":0,"K":0,"M":0,"F":0,"P":0,"S":0,"T":0,"W":0,"Y":0,"V":0}
        for aa in candidate:
            if aa in count_dict.keys():
                count_dict[aa] +=1
        for i in count_dict.values():
            if i > 0.15*len(candidate):
                reps_rules_failed +=1000000000
        for aa in candidate:
            if aa =='-':
                reps_rules_failed +=1000000000
        rules_failed = reps_rules_failed/len(candidate)

        return rules_failed

    def _get_fitness_batch(self, listProts):
        count_batch = []

        for candidate in listProts:
            reps_rules_failed = 0 
            count_dict={"A":0,"R":0,"N":0,"D":0,"C":0,"Q":0,"E":0,"G":0,"H":0,"I":0,"L":0,"K":0,"M":0,"F":0,"P":0,"S":0,"T":0,"W":0,"Y":0,"V":0}
            for aa in candidate:
                if aa in count_dict.keys():
                    count_dict[aa] +=1
            for i in count_dict.values():
                if i > 0.15*len(candidate):
                    reps_rules_failed +=1000000000
            for aa in candidate:
                if aa =='-':
                    reps_rules_failed +=1000000000
            rules_failed = reps_rules_failed/len(candidate)
            count_batch.append(rules_failed)

        return count_batch

    def __str__(self):
        return 'Reps Rules Failed'
        
        

class Max_Hidrophobicity(EvaluationFunction):

    def __init__(self, maximize=True, worst_fitness=-1):
        super(Max_Hidrophobicity, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate):
        h = []
        for a in candidate:
            if a in hydroscale.keys():
                h.append(hydroscale[a])
        hydrophobicity = Average(h)#/len(candidate)

        return hydrophobicity

    def _get_fitness_batch(self, listProts):
        
        hydro_batch =[]
        for candidate in listProts:
            h = []
            for a in candidate:
                if a in hydroscale.keys():
                    h.append(hydroscale[a])
            hydrophobicity = Average(h)#/len(candidate)
            hydro_batch.append(hydrophobicity)
        return hydro_batch


    def __str__(self):
        return 'Maximize hidrophobicity'
        
        
class Prob_Hmm(EvaluationFunction):

    def __init__(self, maximize=True, worst_fitness=0):
        super(Prob_Hmm, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate):
        probability = evaluateHMM(candidate)

        return probability

    def _get_fitness_batch(self, listProts):
        hmm_batch =[]
        for candidate in listProts:
            hmm_batch.append(evaluateHMM(candidate))
        return hmm_batch

    def __str__(self):
        return 'Maximize probability'
