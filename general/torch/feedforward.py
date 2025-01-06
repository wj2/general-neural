import numpy as np
import torch
import torch.nn as nn
import collections

from . import any
from . import regularizers as regs


def _make_network(
    inp,
    layers,
    out,
    transfer_function=nn.ReLU,
    output_function=None,
    weight_init=nn.init.xavier_uniform_,
    **layer_kwargs,
):
    net_modules = [
        ("input", nn.Linear(inp, layers[0], **layer_kwargs)),
        ("hidden_0", transfer_function()),
    ]
    reg_layers = [False, True]
    for i, layer in enumerate(layers[:-1]):
        net_modules.append(
            ("lin_{}".format(i + 1), nn.Linear(layer, layers[i + 1], **layer_kwargs))
        )
        net_modules.append(("hidden_{}".format(i + 1), transfer_function()))
        reg_layers.extend([False, True])
    net_modules.append(("out_lin", nn.Linear(layers[-1], out, **layer_kwargs)))
    if output_function is not None:
        net_modules.append(("out_nonlin", output_function()))
    reg_layers.extend([False, False])
    net = nn.Sequential(collections.OrderedDict(net_modules))
    if weight_init is not None:

        def init_weights(m):
            if isinstance(m, nn.Linear):
                weight_init(m.weight)
                m.bias.data.fill_(0.0)

        net.apply(init_weights)
    return net, reg_layers


class FeedForwardNetwork(any.GenericModule):
    def __init__(
        self,
        input_dim,
        layer_dims,
        output_dim,
        transfer_function=nn.ReLU,
        output_function=nn.Sigmoid,
        rep_ind=-2,
    ):
        super().__init__()
        self.input_size = input_dim
        self.hidden_sizes = layer_dims
        self.output_size = output_dim
        self.rep_ind = rep_ind

        net, reg_layers = _make_network(
            input_dim,
            layer_dims,
            output_dim,
            transfer_function=transfer_function,
            output_function=output_function,
        )
        self.net = net.to(self.device)
        self.reg_layers = reg_layers
        self.use_optimizer = None

    def forward(self, x):
        x_use = self._setup_outsider(x)
        return self.net(x_use)

    def get_representation(self, X, layer=None):
        if layer is None:
            layer = self.rep_ind
        X = self._setup_outsider(X)
        return self.net[:layer](X)

    def get_output(self, X):
        X = self._setup_outsider(X)
        return self.net(X)

    def _setup_outsider(self, inp):
        return torch.tensor(inp).to(self.device).float()

    def fit(
        self,
        X,
        y,
        batch_size=100,
        **kwargs,
    ):
        X = self._setup_outsider(X)
        y = self._setup_outsider(y)
        gen = any.batch_generator(X, y, batch_size=batch_size)
        return self.fit_generator(gen, **kwargs)

    def fit_generator(
        self,
        gen,
        num_steps=100,
        optim=None,
        lr=1e-3,
        loss=None,
        trackers=None,
        reset_optimizer=False,
        l2_reg=0,
        l1_reg=0,
        **kwargs,
    ):
        if trackers is None:
            trackers = []
        if optim is None:
            optim = torch.optim.Adam
        if loss is None:
            loss = nn.MSELoss
        loss_func = loss(**kwargs)
        if reset_optimizer or self.use_optimizer is None:
            self.use_optimizer = optim(self.net.parameters(), lr=lr)
        loss_record = np.zeros(num_steps)
        for i, (X_batch, y_batch) in enumerate(gen):
            X_batch = self._setup_outsider(X_batch)
            y_batch = self._setup_outsider(y_batch)

            self.use_optimizer.zero_grad()
            net_outs = self.net(X_batch)
            loss = loss_func(net_outs, y_batch)
            if l2_reg > 0:
                loss = loss + l2_reg * regs.l2_activity_reg(
                    self.net, X_batch, self.reg_layers
                )
            if l1_reg > 0:
                loss = loss + l1_reg * regs.l1_activity_reg(
                    self.net, X_batch, self.reg_layers
                )
            loss.backward()
            self.use_optimizer.step()
            loss_record[i] = loss.item()
            self.run_trackers(trackers)
            if i >= num_steps - 1:
                break
        out_dict = {}
        out_dict["loss"] = loss_record
        out_dict["trackers"] = self.get_tracker_data(trackers)
        return out_dict

    def run_trackers(self, trackers):
        for tracker in trackers:
            tracker.on_step(self)

    def get_tracker_data(self, trackers):
        out_dict = {}
        for tracker in trackers:
            out_dict[tracker.label] = tracker.finish()
        return out_dict


class AutoEncoder(FeedForwardNetwork):
    def __init__(
        self,
        input_dim,
        layer_dims,
        rep_ind=-1,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            layer_dims,
            input_dim,
            output_function=None,
            rep_ind=rep_ind,
            **kwargs,
        )

    def fit(self, X, batch_size=100, **kwargs):
        X = self._setup_outsider(X)
        gen = any.batch_generator(X, X, batch_size=batch_size)
        return self.fit_generator(gen, **kwargs)
