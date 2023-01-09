from inspyred.ec.emo import Pareto
import numpy as np


class Observers:
    
    def fitness_statistics(self,population):
        """Return the basic statistics of the population's fitness values.
        
        Arguments:
        
        - *population* -- the population of individuals 

        """

        stats = {}
        population.sort(reverse=True)
        first = population[0].fitness
        
        if isinstance(first,Pareto):
            n = len(first.values)
            for i in range(n):
                f = [p.fitness.values[i] for p in population]
                worst_fit = min(f)
                best_fit = max(f)
                med_fit = np.median(f)
                avg_fit = np.mean(f)
                std_fit = np.std(f)
                stats['obj_{}'.format(i)]= {'best': best_fit, 'worst': worst_fit, 'mean': avg_fit,'median': med_fit, 'std': std_fit}    
        else:
            worst_fit = population[-1].fitness
            best_fit = population[0].fitness
            f = [p.fitness for p in population]
            med_fit = np.median(f)
            avg_fit = np.mean(f)
            std_fit = np.std(f)
            stats['obj']= {'best': best_fit, 'worst': worst_fit, 'mean': avg_fit,'median': med_fit, 'std': std_fit}

        return stats 

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    def results_observer(self, population, num_generations, num_evaluations, args):
        """
        Print the output of the evolutionary computation to a file with the follow fields:
        - number of generation
        - fitness of candidate
        - the solution candidates
        - the solution encoded candidates

        Args:
            population (list): the population of Individuals
            num_generations (int): the number of elapsed generations
            num_evaluations (int): the number of evaluations already performed
            args (dict): a dictionary of keyword arguments
        """
        
        stats = self.fitness_statistics(population)
        title = "Gen    Eval|"
        values = "{0:>4} {1:>6}|".format(num_generations,num_evaluations) 
        
        for key in stats:
            s = stats[key]
            title = title +  "     Worst      Best    Median   Average   Std Dev|"
            values = values +  "  {0:.6f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}|".format(s['worst'], 
                                                                                s['best'], 
                                                                                s['median'], 
                                                                                s['mean'], 
                                                                                s['std'])
    
        if num_generations==0:
            print(title)
        print(values)
    
    def __call__(self, population, num_generations, num_evaluations, args):
        stats = self.fitness_statistics(population)
        title = "Gen    Eval|"
        values = "{0:>4} {1:>6}|".format(num_generations,num_evaluations) 
        
        for key in stats:
            s = stats[key]
            title = title +  "     Worst      Best    Median   Average   Std Dev|"
            values = values +  "  {0:.6f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}|".format(s['worst'], 
                                                                                s['best'], 
                                                                                s['median'], 
                                                                                s['mean'], 
                                                                                s['std'])

        print(values)

    def __name__(self):
        return "aaa"