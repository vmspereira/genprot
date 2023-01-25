from genprot.models import VAE 
from genprot.optimization.evaluation import MinRulesSolubility, MinRulesSynthesis, MaxHidrophobicity
from genprot.optimization.problem import ProteinProblem
from genprot.optimization import EA

def run():
    model = VAE(941,20,
              encoder_hidden=[100],
              encoder_dropout=[0.],
              decoder_hidden=[100], 
              decoder_dropout=[0.],
              beta=0.1
              )
    model.load_weights('../output/weights/anti.h5')
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