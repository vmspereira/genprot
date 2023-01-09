from genprot.models import MSAVAE, ARVAE 
from genprot.optimization.evaluation import MinRulesSolubility, MinRulesSynthesis, MaxHidrophobicity
from genprot.optimization.problem import ProteinProblem
from genprot.optimization import EA

def run():
    model = MSAVAE(original_dim=360, latent_dim=10)
    model.load_weights('../output/weights/msavae.h5')
    objectives =[
        MaxHidrophobicity(),
        MinRulesSolubility(),
        MinRulesSynthesis()
    ]
    problem = ProteinProblem(objectives,model=model)
    ea = EA(problem=problem)
    ea.run()


if __name__ == '__main__':
  run()