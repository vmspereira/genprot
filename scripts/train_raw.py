import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from genprot.models.gVAE.vaes import ARVAE
from genprot.utils.io import load_gzdata, read_fasta
from genprot.utils.data_loaders import one_hot_generator

# Define training parameters
batch_size = 32
seed = 0
n_epochs = 50
verbose = 1
save_all_epochs = False
original_dim = 504 #2048
latent_dim = 50 #100
seed = np.random.seed(seed)

_, raw_seqs = read_fasta('data/training_data/seqs_arvae_train.fasta')
_, val_raw_seqs = read_fasta('data/training_data/seqs_arvae_test.fasta')

# Define data generators
train_gen = one_hot_generator(raw_seqs, padding=original_dim)
val_gen = one_hot_generator(val_raw_seqs, padding=original_dim)

# Define model
print('Building model')
model = ARVAE(original_dim=original_dim, latent_dim=latent_dim)
# (Optionally) define callbacks
callbacks=[CSVLogger('output/logs/arvae.csv')]
callbacks.append(EarlyStopping(monitor='loss', patience=3))

if save_all_epochs:
    callbacks.append(ModelCheckpoint('output/weights/arvae'+'.{epoch:02d}-{luxa_errors_mean:.2f}.hdf5',
                                     save_best_only=False, verbose=1))

# Train model https://github.com/keras-team/keras/issues/8595
history = model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(raw_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_raw_seqs) // batch_size,
                        callbacks=callbacks)
 

if not save_all_epochs:
  model.save_weights('../output/weights/arvae.h5')
