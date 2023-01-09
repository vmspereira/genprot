import copy
from inspyred.ec.variators.crossovers import crossover
from inspyred.ec.variators.mutators import mutator



@crossover
def one_point_crossover(random, mom, dad, args):    
    """
        Out of the shelf one point crossover
    """
    crossover_rate = args.setdefault('one_point_crossover_rate', 1.0)
    children = []
    if random.random() < crossover_rate:
        size = len(mom)
        cut_point = random.randint(1,size)
        bro = mom[:cut_point]+dad[cut_point:]
        sis = dad[:cut_point]+mom[cut_point:]
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children



@crossover
def two_point_crossover(random, mom, dad, args):    
    """
        Out of the shelf two point crossover
    """
    crossover_rate = args.setdefault('two_point_crossover_rate', 0.5)
    children = []
    if random.random() < crossover_rate:
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
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


@crossover
def real_arithmetical_crossover(random, mom, dad, args):
    """
        Random trade off of n genes from the progenitors
        The maximum number of trade off is defined by  'num_mix_points'
        For a gene position i and a randmon value a in range 0 to 1
            child_1[i] = a * parent_1[i] + (1-a) * parent_2[i]
            child_2[i] = (1-a) * parent_1[i] + a * parent_2[i]
    """    
    crossover_rate = args.setdefault('real_arithmetical_crossover_rate', 0.5)
    num_mix_points = args.setdefault('num_mix_points', 1)
    children = []
    if random.random() < crossover_rate:
        num_mix = min(len(mom)-1,num_mix_points)
        mix_points = random.sample(range(1, len(mom)), num_mix)
        mix_points.sort()
        bro = copy.copy(dad)
        sis = copy.copy(mom)
        for i, (m, d) in enumerate(zip(mom, dad)):
            if i in mix_points:
                mix = random.random()
                bro[i] = m * mix + d *(1-mix)
                sis[i] = d * mix + m *(1-mix)
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children



@mutator
def gaussian_mutation(random,candidate, args):
    """
        A Gaussian mutator centerd in the gene[i] value
    """
    mut_rate = args.setdefault('gaussian_mutation_rate', 0.1)
    mut_gene_rate = args.setdefault('gaussian_gene_mutation', 0.1)
    mean = args.setdefault('gaussian_mean', 0.0)
    stdev = args.setdefault('gaussian_stdev', 1.0)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    if random.random() < mut_rate:
        for i, m in enumerate(mutant):
            if random.random() < mut_gene_rate:
                mutant[i] += random.gauss(mean, stdev) + m 
        mutant = bounder(mutant, args)
    
    return mutant



@mutator
def single_mutation(random, candidate, args):
    """Returns the mutant produced by a single mutation on the candidate (when the representation is a set of integers).
    The candidate size is maintained.

    Parameters
    ----------
    random  : the random number generator object
    candidate : the candidate solution
    args : a dictionary of keyword arguments

    Returns
    -------
    out : new candidate

    Optional keyword arguments in args:

    - *mutation_rate* -- the rate at which mutation is performed (default 0.1)
    """

    bounder = args["_ec"].bounder
    mutRate = args.setdefault("mutation_rate", 0.1)
    if random.random() > mutRate:
        return candidate
    mutant = copy.copy(candidate)
    index = random.randint(0, len(mutant) - 1) if len(mutant) > 1 else 0
    newElem = bounder.lower_bound + \
                (bounder.upper_bound - bounder.lower_bound) * random.random()
    mutantL = list(mutant)
    mutantL[index] = newElem
    mutant = set(mutantL)
    return mutant
