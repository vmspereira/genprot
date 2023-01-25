'''Models for VAE'''


from keras.layers import Input, Dense, Lambda, Flatten, Dropout, Reshape, Activation
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from functools import partial

from genprot.utils.alphabet import ALPHABET_SIZE
from genprot.utils.data_loaders import to_one_hot, right_pad
from genprot.utils.decoding import *

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sampler(latent_dim, epsilon_std=1):
    _sampling = lambda z_args: (z_args[0] + K.sqrt(tf.convert_to_tensor(z_args[1] + 1e-8, np.float32)) *
                                K.random_normal(shape=K.shape(z_args[0]), mean=0., stddev=epsilon_std))

    return Lambda(_sampling, output_shape=(latent_dim,))

def batch_conds(n_samples, solubility_level):
    target_conds = [0,0,0]
    target_conds[solubility_level] = 1
    target_conds = np.repeat(np.array(target_conds).reshape((1,3)), n_samples, axis=0)
    return target_conds

class VAE:

    def __init__(self,
                seqlen,
                latent_dim,
                alphabet_size = ALPHABET_SIZE, 
                encoder_hidden=[250,250,250],
                encoder_dropout=[0.7,0.,0.], 
                activation='relu', 
                decoder_hidden=[250], 
                decoder_dropout=[0.],
                lr=0.001,
                metrics=['accuracy'],
                beta=0.5):
        '''VAE model.'''
        self.latent_dim = latent_dim
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=(seqlen, alphabet_size,), name='encoder_input')
        x = Flatten()(inputs)
        c = 0
        for n_hid, drop in zip(encoder_hidden, encoder_dropout):
            c +=1
            x = Dense(n_hid, activation=activation)(x)
            if drop > 0:
                x = Dropout(drop)(x)

        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = latent_inputs
        for n_hid, drop in zip(decoder_hidden, decoder_dropout):
            x = Dense(n_hid, activation=activation)(x)
            if drop > 0:
                x = Dropout(drop)(x)

        decoder_out = Dense(seqlen*alphabet_size, activation=None)(x)
        decoder_out = Reshape((seqlen, alphabet_size))(decoder_out)
        outputs = Activation('softmax')(decoder_out)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
    
        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_seq')

        reconstruction_loss = K.sum(categorical_crossentropy(inputs,
                                                outputs),-1)
        reconstruction_loss *= (seqlen*alphabet_size)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -beta
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        vae.compile(optimizer=Adam(learning_rate=lr),metrics=metrics)
        
        self.S = sampler(latent_dim, epsilon_std=1.)
        self.encoder = encoder
        self.decoder = decoder
        self.inputs = inputs
        self.outputs = outputs
        self.vae = vae

    def fit(self,args,**kwargs):
        self.vae.fit(args,**kwargs)
    
    def save_weights(self,args,**kwargs):
        self.vae.save_weights(args,**kwargs)
    
    def load_weights(self,args,**kwargs):
        self.vae.load_weights(args,**kwargs)
        
    def decode(self, z, remove_gaps=False, conditions=None):
        return decode_nonar(self.decoder, z, remove_gaps=remove_gaps, conditions=conditions)
    
    def prior_sample(self, n_samples=1, mean=0, stddev=1,
                     remove_gaps=False, batch_size=5000):
        if n_samples > batch_size:
            x = []
            total = 0
            while total< n_samples:
                this_batch = min(batch_size, n_samples - total)
                z_sample = mean + stddev * np.random.randn(this_batch, self.latent_dim)
                x += self.decode(z_sample, remove_gaps=remove_gaps)
                total += this_batch
        else:
            z_sample = mean + stddev * np.random.randn(n_samples, self.latent_dim)
            x = self.decode(z_sample, remove_gaps=remove_gaps)
        return x

    def generate_variants(self, seq ,num_samples, posterior_var_scale=1., temperature=0.,
                               solubility_level=None):
        
        oh = to_one_hot(right_pad([seq], self.encoder.input_shape[1]))
        oh = np.repeat(oh, num_samples, axis=0)
        orig_conds = np.repeat(np.array([1,0,0]).reshape((1,3)), num_samples, axis=0)
        inputs = oh if solubility_level is None else [oh, orig_conds]

        zmean, zvar, z = self.encoder.predict(inputs)
        print(z.shape)

        if posterior_var_scale != 1.:
            _z = np.sqrt(posterior_var_scale*zvar)*np.random.randn(*zmean.shape) + zmean

        sample_func = None
        if temperature > 0:
            sample_func = partial(batch_temp_sample, temperature=temperature)
        target_conds = None if solubility_level is None else batch_conds(num_samples, solubility_level)
        return self.decode(_z, remove_gaps=True, sample_func=sample_func,
                           conditions=target_conds)