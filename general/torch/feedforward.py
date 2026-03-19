import math
import torch
import torch.nn as nn
import collections

from . import any


def make_feedforward_network(
    inp,
    layers,
    out,
    transfer_function=nn.ReLU,
    output_function=None,
    weight_init=nn.init.xavier_uniform_,
    dropout=0,
    **layer_kwargs,
):
    most_recent_size = inp
    net_modules = []
    reg_layers = []
    if len(layers) > 0:
        net_modules.append(
            ("input", nn.Linear(most_recent_size, layers[0], **layer_kwargs))
        )
        net_modules.append(("hidden_0", transfer_function()))
        reg_layers.extend([False, True])
        most_recent_size = layers[0]
    for i, layer in enumerate(layers[1:]):
        net_modules.append(
            (
                "lin_{}".format(i + 1),
                nn.Linear(most_recent_size, layer, **layer_kwargs),
            )
        )
        net_modules.append(("hidden_{}".format(i + 1), transfer_function()))
        reg_layers.extend([False, True])
        most_recent_size = layer
        if dropout > 0:
            net_modules.append(("drop_{}".format(i + 1), nn.Dropout(dropout)))
            reg_layers.append(False)

    net_modules.append(("out_lin", nn.Linear(most_recent_size, out, **layer_kwargs)))
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


class FeedForwardNetwork(any.GenericModule, any.GenericTrainingLoop):
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

        net, reg_layers = make_feedforward_network(
            input_dim,
            layer_dims,
            output_dim,
            transfer_function=transfer_function,
            output_function=output_function,
        )
        self.net = net.to(self.device)
        self.reg_layers = reg_layers
        self.use_optimizer = None

    def get_representation(self, X, layer=None):
        if layer is None:
            layer = self.rep_ind
        X = self._setup_outsider(X)
        return self.net[:layer](X)


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class FullAttentionLayer(nn.Module):
    def __init__(
        self, attn_dim, ff_dim=None, n_heads=1, batch_first=True, dropout=0.1, **kwargs
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = attn_dim
        self.attn = nn.MultiheadAttention(
            attn_dim,
            n_heads,
            batch_first=batch_first,
            **kwargs,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.ff1 = nn.Linear(attn_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, attn_dim)
        self.relu1 = nn.ReLU()

    def forward(self, x1, x2=None, **kwargs):
        if x2 is None:
            x2 = x1
        x = self.dropout1(self.attn(x1, x2, x2, **kwargs)[0])
        x = self.norm1(x1 + x)
        x = self.norm2(x + self.dropout2(self.ff2(self.relu1(self.ff1(x)))))
        return x


class ModuleAttention(nn.Module):
    def __init__(
        self,
        n_modules,
        module_dims,
        attn_dim,
        output_dim,
        n_heads=1,
        output_function=nn.Softmax,
        include_positional=False,
        dropout=0.1,
    ):
        super().__init__()
        assert len(module_dims) == n_modules
        self.n_modules = n_modules

        if include_positional:
            self.positional = PositionalEncoding(attn_dim, dropout=dropout)
        self.include_positional = include_positional
        self.classifier_tokens = nn.Parameter(torch.zeros(n_modules, 1, 1, attn_dim)) 
        self.transform_layers = list(nn.Linear(md, attn_dim) for md in module_dims)
        list(
            self.add_module("inp_{}".format(i), x)
            for i, x in enumerate(self.transform_layers)
        )
        self.self_attn_layers = list(
            FullAttentionLayer(attn_dim, n_heads, batch_first=True, dropout=dropout)
            for _ in module_dims
        )
        list(
            self.add_module("self_{}".format(i), x)
            for i, x in enumerate(self.self_attn_layers)
        )
        cross_dim = attn_dim * (n_modules - 1)
        self.cross_attn_layers = list(
            FullAttentionLayer(
                attn_dim,
                n_heads,
                vdim=cross_dim,
                kdim=cross_dim,
                batch_first=True,
                dropout=dropout,
            )
            for _ in module_dims
        )
        list(
            self.add_module("cross_{}".format(i), x)
            for i, x in enumerate(self.cross_attn_layers)
        )
        self.out_lin = nn.Linear(attn_dim, output_dim)
        self.out_trs = output_function(dim=-1)

    def forward(self, Xs, mask=None, **kwargs):
        reps = []
        if mask is None:
            mask = (None,) * self.n_modules
        for i, X in enumerate(Xs):
            ri = self.transform_layers[i](X)
            if self.include_positional:
                ri = self.positional(ri)
            ct = self.classifier_tokens[i]
            use_ct = ct.expand(len(ri), -1, -1)
            ri = torch.concatenate((use_ct, ri), axis=1)
            reps.append(
                self.self_attn_layers[i](
                    ri,
                    ri,
                    key_padding_mask=mask[i],
                    **kwargs,
                )
            )
        cross_reps = []
        for i, ri in enumerate(reps):
            alt_rj = torch.concatenate(reps[:i] + reps[i + 1 :], axis=-1)
            r_ij = self.cross_attn_layers[i](
                ri,
                alt_rj,
                key_padding_mask=mask[i],
                **kwargs,
            )
            cross_reps.append(r_ij)
        combined = torch.mean(torch.stack(cross_reps, dim=0), dim=0)
        out = self.out_trs(self.out_lin(combined))
        return out[:, 1:]


class SimpleAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        transform_layer_dim,
        output_dim,
        n_heads=1,
        output_function=nn.Softmax,
        is_causal=True,
        dropout=0.1,
        include_positional=False,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.transform_layer = nn.Linear(
            input_dim,
            transform_layer_dim,
        )
        if include_positional:
            self.positional = PositionalEncoding(transform_layer_dim, dropout=dropout)
        self.include_positional = include_positional
        self.main_net = FullAttentionLayer(
            transform_layer_dim,
            n_heads=n_heads,
            batch_first=True,
            dropout=dropout,
            **kwargs,
        )
        self.out_lin = nn.Linear(transform_layer_dim, output_dim)
        self.out_trs = output_function(dim=-1)
        self.is_causal = is_causal
        self.device = device

    def forward(self, X, mask=None, **kwargs):
        if self.is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(X.shape[1])
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None
        X = self.transform_layer(X)
        if self.include_positional:
            X = self.positional(X)
        output = self.main_net(
            X,
            X,
            is_causal=self.is_causal,
            attn_mask=attn_mask,
            key_padding_mask=mask,
            **kwargs,
        )
        return self.out_trs(self.out_lin(output))


class ModuleAttentionNetwork(any.MultimodalModule, any.MultimodalTrainingLoop):
    def __init__(
        self,
        n_modules,
        module_dims,
        attn_dim,
        output_dim,
        n_heads=1,
        output_function=nn.Softmax,
        **kwargs,
    ):
        super().__init__()
        self.module_sizes = module_dims
        self.n_modules = n_modules
        self.attn_dim = attn_dim
        self.output_size = output_dim
        net = ModuleAttention(
            n_modules,
            module_dims,
            attn_dim,
            output_dim,
            n_heads=n_heads,
            output_function=output_function,
            **kwargs,
        )
        self.net = net.to(self.device)
        self.use_optimizer = None


class AttentionNetwork(any.GenericModule, any.GenericTrainingLoop):
    def __init__(
        self,
        input_dim,
        output_dim,
        transform_layer_dim=500,
        n_heads=1,
        output_function=nn.Softmax,
        is_causal=True,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        net = SimpleAttention(
            input_dim,
            transform_layer_dim,
            output_dim,
            n_heads=n_heads,
            output_function=output_function,
            is_causal=is_causal,
            device=self.device,
            **kwargs,
        )
        self.net = net.to(self.device)
        self.use_optimizer = None
