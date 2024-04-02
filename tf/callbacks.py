
import numpy as np
import tensorflow as tf

import general.utility as u

tfk = tf.keras
tfkl = tf.keras.layers


class TrackWeights(tfk.callbacks.Callback):
    def __init__(self, model, layer_ind, *args, **kwargs):
        self.model = model
        self.layer_ind = layer_ind
        super().__init__(*args, **kwargs)
        self.weights = []

    def on_train_begin(self, logs=None):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.weights
        self.weights.append(np.array(weights[self.layer_ind]))


class TrackReps(tfk.callbacks.Callback):
    def __init__(self, model, *args, n_rep_samps=10**4, mean_tasks=True,
                 only_groups=None, sample_all=False, **kwargs):
        self.modu_model = model
        super().__init__(*args, **kwargs)

        if sample_all:
            stim, inp_rep, targ = model.get_all_stim()
        else:
            stim, inp_rep, targ = model.sample_reps(n_rep_samps)
        self.inp_rep = inp_rep
        self.stim = stim
        self.targ = targ

    def on_train_begin(self, logs=None):
        self.reps = []
        self.reps.append(self.modu_model.get_representation(self.inp_rep))

    def on_epoch_end(self, epoch, logs=None):
        self.reps.append(self.modu_model.get_representation(self.inp_rep))


class CorrCallback(tfk.callbacks.Callback):
    def __init__(self, model, inp, targ, name, *args, loss=None, **kwargs):
        self.modu_model = model
        if loss is None:
            loss = model.loss
        self.loss = loss
        self.inp = inp
        self.targ = targ
        self.name = name
        self.losses = []
        self.resps = []
        super().__init__(*args, **kwargs)

    def _compute_corr(self):
        m_out = self.modu_model.model(self.inp).numpy()
        loss = self.loss(self.targ, m_out)
        return m_out, loss
        
    def on_train_begin(self, logs=None):
        self.resps = []
        self.losses = []

        r, t_loss = self._compute_corr()
        self.resps.append(r)
        self.losses.append(t_loss)

    def on_epoch_end(self, epoch, logs=None):
        r, t_loss = self._compute_corr()
        self.resps.append(r)
        self.losses.append(t_loss)

    def on_train_end(self, logs=None):
        self.losses = np.array(self.losses)
        self.resps = np.array(self.resps)
        

class DimCorrCallback(tfk.callbacks.Callback):
    def __init__(self, model, *args, dim_samps=10**4, mean_tasks=True,
                 only_groups=None, sample_kwargs=None, **kwargs):
        self.modu_model = model
        super().__init__(*args, **kwargs)
        self.dim = []
        self.corr = []
        self.dim_samps = dim_samps
        self.mean_tasks = mean_tasks
        self.only_groups = only_groups
        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs

    def on_train_begin(self, logs=None):
        self.dim = []
        self.dim_c0 = []
        self.corr = []

        _, _, reps = self.modu_model.sample_reps(self.dim_samps)
        dim = u.participation_ratio(reps)

        _, _, reps_c0 = self.modu_model.sample_reps(self.dim_samps, **self.sample_kwargs)
        dim_c0 = u.participation_ratio(reps)
        corr = 1 - self.modu_model.get_ablated_loss(mean_tasks=self.mean_tasks,
                                                    only_groups=self.only_groups)

        self.dim.append(dim)
        self.dim_c0.append(dim_c0)
        self.corr.append(corr)

    def on_epoch_end(self, epoch, logs=None):
        _, _, reps = self.modu_model.sample_reps(self.dim_samps)
        dim = u.participation_ratio(reps)

        _, _, reps_c0 = self.modu_model.sample_reps(self.dim_samps, **self.sample_kwargs)
        dim_c0 = u.participation_ratio(reps_c0)

        corr = 1 - self.modu_model.get_ablated_loss(mean_tasks=self.mean_tasks,
                                                    only_groups=self.only_groups)
        self.dim.append(dim)
        self.dim_c0.append(dim_c0)
        self.corr.append(corr)

    def on_train_end(self, logs=None):
        self.dim = np.array(self.dim)
        self.dim_c0 = np.array(self.dim_c0)
        self.corr = np.array(self.corr)
