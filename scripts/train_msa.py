import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from genprot.utils.io import load_gzdata, read_fasta
from genprot.utils.data_loaders import one_hot_generator, to_one_hot
from genprot.models.VAE.model import VAE
# Required in M1 metal
disable_eager_execution()


def train():
  # Define training parameters
  batch_size = 32
  seed = 0
  n_epochs = 20
  verbose = 1
  save_all_epochs = False

  seed = np.random.seed(seed)

  # Load aligned sequences
  #_, msa_seqs = read_fasta('../data/training_data/seqs_msavae_train.fasta')
  #_, val_msa_seqs = read_fasta('../data/training_data/seqs_msavae_test.fasta')

  _, msa_seqs = read_fasta('../data/datasets/anti/train.fasta')
  _, val_msa_seqs = read_fasta('../data/datasets/anti/test.fasta')

  train_gen = one_hot_generator(msa_seqs, padding=None)
  val_gen = one_hot_generator(val_msa_seqs, padding=None)

  # X_train = to_one_hot(msa_seqs)
  # X_val = to_one_hot(val_msa_seqs)

  # Define model
  print('Building model')
  #model = MSAVAE(original_dim=941, latent_dim=50)
  
  model = VAE(941,20,
              encoder_hidden=[100],
              encoder_dropout=[0.],
              decoder_hidden=[100], 
              decoder_dropout=[0.],
              beta=0.5
              )
  
  model.vae.summary()
  
  callbacks=[CSVLogger('../output/logs/vae.csv')]
  callbacks.append(EarlyStopping(monitor='loss', patience=3))
  if save_all_epochs:
      callbacks.append(ModelCheckpoint('../output/weights/vae'+'.{epoch:02d}}.hdf5',
                                      save_best_only=False, verbose=1))
  
  model.fit(train_gen,
          epochs=n_epochs,
          steps_per_epoch=len(msa_seqs) // batch_size,
          validation_data=val_gen,
          validation_steps=len(val_msa_seqs) // batch_size,
          callbacks=callbacks)

  model.save_weights('../output/weights/anti.h5')
  
  x = model.prior_sample(remove_gaps=True)
  # antibody VHH and VL are separated by '&'
  VH, VL = tuple(x[0].split('&'))
  print(VH,VL)

if __name__ == '__main__':
  train()