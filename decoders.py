
import numpy as np
import pickle
import sklearn.kernel_approximation as skka
import pystan as ps
import arviz as az
import functools as ft
import sklearn.pipeline as sklpipe
import sklearn.preprocessing as skp
import sklearn.decomposition as skd

import general.utility as u
import general.stan_utility as su

import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

periodic_decoder_path = 'general/stan_decoder/von_mises_pop.pkl'
periodic_decoder_time_path = 'general/stan_decoder/von_mises_poptime.pkl'
pd_arviz = {'observed_data':'y',
            'log_likelihood':{'y':'log_lik'},
            'posterior_predictive':'err_hat'}

class PeriodicDecoder:

    def __init__(self, C=1, epsilon=1, kernel='rbf', 
                 include_x=False, **kernel_params):
        self.c = C
        self.epsilon = epsilon
        self._fit = None
        self.include_x = include_x
        if kernel == 'rbf':
            self.kernel_params = kernel_params
            if self.kernel_params.get('gamma', None) is None:
                self.kernel_params['gamma'] = 'scale'
            self.kernel = skka.RBFSampler

    def kernel_transform(self, x):
        x_kern = self.preprocessing.transform(x)
        if self.include_x:
            x_kern = np.concatenate((x, x_kern), axis=1)
        return x_kern

    """ why does normalizing make the model so ill-posed? """ 
    def kernel_fit(self, x, norm=True, repca=None):
        if self.kernel_params['gamma'] == 'scale':
            self.kernel_params['gamma'] = 1/(x.shape[1]*x.var())
        # self.kernel = self.kernel(**self.kernel_params)
        # self.kernel = self.kernel.fit(x)
        preproc = [self.kernel(**self.kernel_params)]
        if norm:
            preproc.append(skp.StandardScaler())
        if repca is not None:
            preproc.append(skd.PCA(repca))
        pipe = sklpipe.make_pipeline(*preproc)
        self.preprocessing = pipe.fit(x)

    def fit(self, x, y, *args):
        pass
    
    def get_fit(self):
        if self._fit is not None:
            f = self._fit
        else:
            raise IOError('model has not been fit')
        return f
    
    def predict(self, x, ct_func=None):
        pass

    def score(self, x, y, *args, ct_func=np.mean, norm=True):
        y_hat = self.predict(x, *args)
        y_diff = u.normalize_periodic_range(y_hat - np.expand_dims(y, 0))
        score = np.sum(y_diff**2, axis=1)
        if norm:
            rand = np.sum(u.normalize_periodic_range(y)**2)
            score = 1 - score/rand
        return score

negloglik = lambda y, rv_y: -rv_y.log_prob(y)
class PeriodicDecoderTF(PeriodicDecoder):

    def __init__(self, *args, epochs=1000, verbose=False, learning_rate=.01,
                 **kwargs):
        self.epochs = epochs
        self.verbose = verbose
        self.learning_rate = learning_rate
        super().__init__(*args, **kwargs)

    def make_decoding_model(self, x_inp):
        layers = []
        layers.append(tfkl.InputLayer(input_shape=x_inp.shape[1]))
        
        reg = tfk.regularizers.l2(1/(100*self.c))
        layers.append(tfkl.Dense(2, kernel_regularizer=reg))
        distr_func = lambda t: tfd.VonMises(t[..., :1], t[..., 1:])
        layers.append(tfp.layers.DistributionLambda(distr_func))
        model = tfk.Sequential(layers)
        return model
        
    def fit(self, x, y, *args):
        self.kernel_fit(x)
        x_kern = self.kernel_transform(x)
        model = self.make_decoding_model(x_kern)
        opt = tf.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt,
                      loss=negloglik)
        model.fit(x_kern, y, epochs=self.epochs, verbose=self.verbose);
        self._fit = model
        return self

    def predict(self, x, mean=True):
        x_kern = self.kernel_transform(x)
        distr = self._fit(x_kern)
        if mean:
            out = distr.mean()
        else:
            out = distr.sample()
        return out

class PeriodicDecoderStan:

    def __init__(self, C=1, epsilon=1, kernel='rbf', decoder='stan',
                 stan_path=periodic_decoder_path, recompile_decoder=False,
                 adapt_delta=.9, max_treedepth=15, iters=800, chains=4,
                 trans_func=None, az_manifest=None, include_x=False,
                 **kernel_params):
        if az_manifest is None:
            self.manifest = pd_arviz
        self.c = C
        self.epsilon = epsilon
        self.iters = iters
        self.chains = chains
        self.trans_func = trans_func
        self._fit = None
        self.include_x = include_x
        if kernel == 'rbf':
            self.kernel_params = kernel_params
            if self.kernel_params.get('gamma', None) is None:
                self.kernel_params['gamma'] = 'scale'
            self.kernel = skka.RBFSampler
        if decoder == 'stan':
            if recompile_decoder:
                su.recompile_model(stan_path)
            self.decoder_model = pickle.load(open(stan_path, 'rb'))
            self.decoder_control = {'adapt_delta':adapt_delta,
                                    'max_treedepth':max_treedepth}

    def kernel_transform(self, x):
        x_kern = self.preprocessing.transform(x)
        if self.include_x:
            x_kern = np.concatenate((x, x_kern), axis=1)
        return x_kern

    """ why does normalizing make the model so ill-posed? """ 
    def kernel_fit(self, x, norm=True, repca=None):
        if self.kernel_params['gamma'] == 'scale':
            self.kernel_params['gamma'] = 1/(x.shape[1]*x.var())
        # self.kernel = self.kernel(**self.kernel_params)
        # self.kernel = self.kernel.fit(x)
        preproc = [self.kernel(**self.kernel_params)]
        if norm:
            preproc.append(skp.StandardScaler())
        if repca is not None:
            preproc.append(skd.PCA(repca))
        pipe = sklpipe.make_pipeline(*preproc)
        self.preprocessing = pipe.fit(x)

    def _make_stan_dict(self, x, y):
        stan_data = dict(x=x, N=x.shape[0], K=x.shape[1], y=y,
                         beta_var=self.c, sigma_var=1/self.epsilon,
                         sigma_mean=1/self.epsilon)
        return stan_data
    
    def fit(self, x, y, *args):
        self.kernel_fit(x)
        x_kern = self.kernel_transform(x)
        stan_data = self._make_stan_dict(x_kern, y, *args)
        _fit = self.decoder_model.sampling(data=stan_data, iter=self.iters,
                                            chains=self.chains,
                                            control=self.decoder_control)
        _diag = ps.diagnostics.check_hmc_diagnostics(_fit)
        self._fit = _fit
        self.diag_x = _diag
        return self

    def get_fit(self):
        if self._fit is not None:
            f = self._fit
        else:
            raise IOError('model has not been fit')
        return f

    def get_arviz(self):
        f = self.get_fit()
        out = az.from_pystan(posterior=f, **self.manifest)
        return out
    
    def predict(self, x, ct_func=None):
        if ct_func is None and self.trans_func is not None:
            ct_func = ft.partial(self.trans_func, axis=0)
        else:
            ct_func = lambda x: x
        x_kern = self.kernel_transform(x)
        trans = ct_func(self._fit.extract()['beta'])
        offset = ct_func(self._fit.extract()['b'])
        y = np.expand_dims(offset, 1) + np.dot(trans, x_kern.T)
        y = np.mod(y, 2*np.pi)
        return y

    def score(self, x, y, *args, ct_func=np.mean, norm=True):
        y_hat = self.predict(x, *args)
        y_diff = u.normalize_periodic_range(y_hat - np.expand_dims(y, 0))
        score = np.sum(y_diff**2, axis=1)
        if norm:
            rand = np.sum(u.normalize_periodic_range(y)**2)
            score = 1 - score/rand
        return score
        
class PeriodicDecoderStanTime(PeriodicDecoderStan):

    def __init__(self, *args, stan_path=periodic_decoder_time_path, **kwargs):
        super().__init__(*args, stan_path=stan_path, **kwargs)

    def _make_stan_dict(self, x, y, t):
        big_t = len(np.unique(t))
        stan_data = dict(x=x, N=x.shape[0], K=x.shape[1], T=big_t,
                         time=t, y=y,
                         beta_mean_var=self.c, beta_var_mean=self.c,
                         beta_var_var=self.c,
                         sigma_mean_mean=1/self.epsilon,
                         sigma_mean_var=1/self.epsilon,
                         sigma_var_mean=1/self.epsilon,
                         sigma_var_var=1/self.epsilon)
        return stan_data

    def predict(self, x, ts, ct_func=None):
        if ct_func is None and self.trans_func is not None:
            ct_func = ft.partial(self.trans_func, axis=0)
        else:
            ct_func = lambda x: x
        x_kern = self.kernel_transform(x)
        trans = ct_func(self._fit.extract()['beta'][:, ts - 1])
        offset = ct_func(self._fit.extract()['b'][:, ts - 1])
        trans_trls = np.sum(np.expand_dims(x_kern, 0)*trans, axis=2)
        y = offset + trans_trls
        y = np.mod(y, 2*np.pi)
        return y
