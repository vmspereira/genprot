import pickle
from genprot.models.gVAE.vaes import MSAVAE, ARVAE
from genprot.utils.io import output_fasta


def main(weights_file, msa=True, num_samples=3000, output_file=None, model_kwargs=None):

  if output_file is None:
    base_name = weights_file.split('/')[-1].split('.')[0]
    output_file = '../output/generated_sequences/{}_samples.fa'.format(base_name)

  if model_kwargs is None:
    model_kwargs = {}
  else:
    with open(model_kwargs, 'rb') as p:
      model_kwargs = pickle.load(p)

  if msa:
    model = MSAVAE(original_dim=360, latent_dim=10)
  else:
    model = ARVAE(**model_kwargs)

  model.load_weights(weights_file)

  samples = model.prior_sample(num_samples)
  names = ['s{}'.format(i+1) for i in range(num_samples)]
  output_fasta(names, samples, output_file)


if __name__ == '__main__':
  main('../output/weights/msavae.h5', msa=True, num_samples=10000,
       output_file=None, model_kwargs=None)