
import numpy as np
import scipy.stats as sts
import tensorflow as tf

import general.tf.callbacks as gtfc
import general.tf.losses as gtfl
import general.tasks.classification as gtc

tfk = tf.keras
tfkl = tf.keras.layers


def train_network(
        model, 
        train_x,
        train_targ,
        epochs=15,
        batch_size=100,
        use_early_stopping=True,
        es_patience=2,
        es_field="val_loss",
        track_mean_tasks=True,
        track_dimensionality=False,
        track_reps=True,
        val_only_groups=None,
        **kwargs,
):
    
    if use_early_stopping:
        cb = tfk.callbacks.EarlyStopping(
            monitor=es_field, mode="min", patience=es_patience,
        )
        curr_cb = kwargs.get("callbacks", [])
        curr_cb.append(cb)
        kwargs["callbacks"] = curr_cb
    if track_dimensionality:
        cb = kwargs.get("callbacks", [])
        d_callback = gtfc.DimCorrCallback(model, mean_tasks=track_mean_tasks,
                                     only_groups=val_only_groups)
        cb.append(d_callback)
        kwargs["callbacks"] = cb
    if track_reps:
        cb = kwargs.get("callbacks", [])
        try:
            rep_callback = gtfc.TrackReps(model, sample_all=True)
        except AttributeError:
            rep_callback = gtfc.TrackReps(model, n_rep_samps=2000)
        cb.append(rep_callback)
        kwargs["callbacks"] = cb

    out = model.model.fit(
        x=train_x,
        y=train_targ,
        epochs=epochs,
        batch_size=batch_size,
        **kwargs
    )
    if track_dimensionality:
        out.history["dimensionality"] = d_callback.dim
        out.history["dimensionality_c0"] = d_callback.dim_c0
        out.history["corr_rate"] = d_callback.corr
    if track_reps:
        out.history["tracked_activity"] = (
            rep_callback.stim,
            rep_callback.inp_rep,
            rep_callback.targ,
            np.stack(rep_callback.reps, axis=0)
        )
    return out


def make_ff_network(
        inp,
        rep,
        out,
        act_func=tf.nn.relu,
        layer_type=tfkl.Dense,
        out_act=tf.nn.sigmoid,
        hidden=(),
        hidden_same_reg=True,
        noise=0.1,
        inp_noise=0.01,
        kernel_reg_type=tfk.regularizers.L2,
        kernel_reg_weight=0,
        act_reg_type=tfk.regularizers.l2,
        act_reg_weight=0,
        constant_init=None,
        kernel_init=None,
        out_kernel_init=None,
        out_constant_init=None,
        use_bias=True,
        **layer_params
):
    layer_list = []
    layer_list.append(tfkl.InputLayer(input_shape=inp))
    if kernel_init is not None:
        rep_kernel_init = tfk.initializers.RandomNormal(stddev=kernel_init)
        hidden_kernel_inits = list(tfk.initializers.RandomNormal(stddev=kernel_init)
                                  for _ in hidden)
    elif constant_init is not None:
        rep_kernel_init = tfk.initializers.Constant(constant_init)
        hidden_kernel_inits = list(tfk.initializers.Constant(constant_init)
                                   for _ in hidden)
    else:
        rep_kernel_init = tfk.initializers.GlorotUniform()
        hidden_kernel_inits = list(tfk.initializers.GlorotUniform() for _ in hidden)
    if out_kernel_init is not None:
        out_kernel_init = tfk.initializers.RandomNormal(stddev=out_kernel_init)
    elif out_constant_init is not None:
        out_kernel_init = tfk.initializers.Constant(constant_init)
    else:
        out_kernel_init = tfk.initializers.GlorotUniform()

    if inp_noise > 0:
        layer_list.append(tfkl.GaussianNoise(inp_noise))
    if kernel_reg_weight > 0:
        kernel_reg = kernel_reg_type(kernel_reg_weight)
    else:
        kernel_reg = None
    if act_reg_weight > 0:
        act_reg = act_reg_type(act_reg_weight)
    else:
        act_reg = None
    if hidden_same_reg:
        use_ah = dict(
            kernel_regularizer=kernel_reg,
            activity_regularizer=act_reg,
            use_bias=use_bias,
        )
    else:
        use_ah = dict()
    use_ah.update(layer_params)

    models = []
    for i, ah in enumerate(hidden):
        lh_ah = layer_type(
            ah, activation=act_func, kernel_initializer=hidden_kernel_inits[i], **use_ah,
        )
        layer_list.append(lh_ah)
        models.append(tfk.Sequential(layer_list))

    lh = layer_type(
        rep,
        activation=act_func,
        kernel_regularizer=kernel_reg,
        activity_regularizer=act_reg,
        kernel_initializer=rep_kernel_init,
        use_bias=use_bias,
        **layer_params
    )
    layer_list.append(lh)
    if noise > 0:
        layer_list.append(tfkl.GaussianNoise(noise))
        
    rep = tfk.Sequential(layer_list)
    models.append(rep)
    layer_list.append(
        tfkl.Dense(
            out,
            activation=out_act,
            kernel_regularizer=kernel_reg,
            kernel_initializer=out_kernel_init,
            use_bias=use_bias,
        )
    )
    enc = tfk.Sequential(layer_list)
    rep_out = tfk.Sequential(layer_list[-1:])
    return enc, models, rep, rep_out


class GenericFFNetwork:
    def __init__(
            self, input_generator, n_rep, tasks=None, noise=.1, **kwargs,
    ):
        self.input_generator = input_generator
        self.tasks = tasks
        out = make_ff_network(
            input_generator.output_dim, n_rep, len(tasks), noise=noise, **kwargs,
        )
        self.layer_reps = out[1]
        self.rep_model = out[2]
        self.model = out[0]
        self.rep_to_out = out[3]
        self.compiled = False
        self.n_layers = len(self.layer_reps)
        self.noise_std = noise
        
    def _compile(self, optimizer=None, loss=None, ignore_nan=True,
                 lr=1e-3):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=lr)
        if loss is None:
            if ignore_nan:
                loss = gtfl.mse_nanloss
            else:
                loss = tf.losses.MeanSquaredError()
        self.model.compile(optimizer, loss)
        self.loss = loss
        self.compiled = True

    def sample_stim(self, n_samps):
        stim, _ = self.input_generator.sample_reps(n_samps)
        return stim

    def get_layer_representation(self, stim, layer=-1, add_noise=False):
        rep = self.layer_reps[layer](stim)
        if add_noise:
            rep = rep + sts.norm(0, self.noise_std).rvs(rep.shape)
        return rep

    def get_representation(self, stim, layer=-1, **kwargs):
        rep = self.get_layer_representation(stim, layer=layer, **kwargs)
        return rep

    def sample_reps(self, n_samps=1000, layer=-1, **kwargs):
        stim, input_rep = self.input_generator.sample_reps(n_samps)
        model_rep = self.get_representation(input_rep, layer=layer, **kwargs)
        return stim, input_rep, model_rep

    def sample_layer_reps(self, n_samps=1000, **kwargs):
        stim, input_rep = self.input_generator.sample_reps(n_samps)
        lrs = tuple(
            self.get_layer_representation(input_rep, layer=i, **kwargs)
            for i in range(self.n_layers)
        )
        return stim, input_rep, lrs

    def get_target(self, inp):
        return self.tasks(inp)

    def sample_targs(self, n_samps=1000):
        stim, input_rep = self.input_generator.sample_reps(n_samps)
        targ = self.get_target(stim)
        return stim, input_rep, targ

    def fit(
        self,
        n_train=2 * 10**4,
        epochs=20,
        batch_size=200,
        n_val=10**3,
        verbose=False,
        **kwargs
    ):
        if not self.compiled:
            self._compile()

        _, x, y = self.sample_targs(n_train)
        _, eval_x, eval_y = self.sample_targs(n_val)

        val_set = (eval_x, eval_y)

        out = train_network(
            self,
            x,
            y,
            epochs=epochs,
            validation_data=val_set,
            verbose=verbose,
            **kwargs,
        )
        return out
    
    
class IdentityNetwork(GenericFFNetwork):
    def __init__(self, input_generator, *args, tasks=None, **kwargs):
        def identity(x):
            return x
        if tasks is None:
            tasks = gtc.IdentityTask(1)
        super().__init__(input_generator, *args, tasks=tasks, **kwargs)
        self.layer_reps = list(identity for _ in self.layer_reps)
        self.rep_model = identity
        self.rep_to_out = identity

    def fit(self, *args, **kwargs):
        return None
