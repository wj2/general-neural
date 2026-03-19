import numpy as np

import torch
import torch.nn as nn
import ragged

from . import regularizers as regs


class GenericModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def get_output(self, X, remake_nested=True, return_mask=False, **kwargs):
        X = self._setup_outsider(X)
        was_nested = False
        use_mask = None
        ret_mask = None
        if X.is_nested:
            offsets = X.offsets()
            X = torch.nested.to_padded_tensor(X, -1000)
            use_mask = (X == -1000)[..., 0]
            was_nested = True
            ret_mask = torch.logical_not(use_mask)
            use_kwargs = {**kwargs}
            use_kwargs["mask"] = use_mask
        else:
            use_kwargs = kwargs

        out = self.net(X, **use_kwargs)
        if was_nested and remake_nested:
            seq_lens = offsets.diff()
            out = torch.nested.narrow(
                out, dim=1, start=0, length=seq_lens, layout=torch.jagged
            )
        if return_mask:
            out = (out, ret_mask)
        return out

    def get_loss(self, X, y, loss, **kwargs):
        pred, mask = self.get_output(X, return_mask=True, remake_nested=False, **kwargs)
        if mask is not None:
            mask = torch.stack((mask,) * y.shape[-1], dim=-1)
            if y.is_nested:
                y = torch.nested.to_padded_tensor(y, -1000)
            out = loss(pred[mask], y[mask])
        else:
            out = loss(pred, y)
        return out

    def get_detached_output(self, X):
        return self._make_numpy(self.get_output(X))

    def _make_numpy(self, X):
        if len(X.shape) == 3:
            X = X.swapaxes(1, 2)
        if X.is_nested:
            out = ragged.array(list(el.detach().cpu().numpy() for el in X))
        else:
            out = X.detach().cpu().numpy()
        return out

    def _setup_outsider(self, inp):
        if isinstance(inp, torch.Tensor):
            out = inp.detach().clone().type(torch.float).to(self.device)
        else:
            out = torch.tensor(inp, dtype=torch.float).to(self.device)
        return out

    def forward(self, x):
        x_use = self._setup_outsider(x)
        return self.net(x_use)

    def run_trackers(self, trackers):
        self.net.eval()
        for tracker in trackers:
            tracker.on_step(self)

    def get_tracker_data(self, trackers):
        out_dict = {}
        for tracker in trackers:
            out_dict[tracker.label] = tracker.finish()
        return out_dict


class MultimodalModule(GenericModule):
    def get_output(self, X, remake_nested=True, return_mask=False, **kwargs):
        X = list(self._setup_outsider(xi) for xi in X)
        was_nested = False
        use_mask = None
        ret_mask = None
        if X[0].is_nested:
            was_nested = True
            X_new = []
            for xi in X:
                offsets = xi.offsets()
                xi = torch.nested.to_padded_tensor(xi, -1000)
                use_mask = (xi == -1000)[..., 0]
                ret_mask = torch.logical_not(use_mask)
                X_new.append(xi)
            X = X_new

        out = self.net(X, mask=use_mask, **kwargs)
        if was_nested and remake_nested:
            seq_lens = offsets.diff()
            out = torch.nested.narrow(
                out, dim=1, start=0, length=seq_lens, layout=torch.jagged
            )
        if return_mask:
            out = (out, ret_mask)
        return out


class MultimodalTrainingLoop:
    def fit(
        self,
        X,
        y,
        batch_size=100,
        ragged=False,
        **kwargs,
    ):
        X = list(self._setup_outsider(xi) for xi in X)
        y = self._setup_outsider(y)
        gen = multimodal_batch_generator(X, y, ragged=ragged, batch_size=batch_size)
        return self.fit_generator(gen, **kwargs)

    def fit_generator(
        self,
        gen,
        num_steps=None,
        optim=None,
        lr=1e-3,
        loss=None,
        trackers=None,
        reset_optimizer=False,
        l2_reg=0,
        l1_reg=0,
        num_epochs=10,
        lr_stepping=False,
        lr_patience=0.1,
        val_set=None,
        forward_kwargs=None,
        optim_kwargs=None,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = len(gen)
        if trackers is None:
            trackers = []
        if optim is None:
            optim = torch.optim.Adam
        if optim_kwargs is None:
            optim_kwargs = {}
        if loss is None:
            loss = nn.MSELoss
        if forward_kwargs is None:
            forward_kwargs = {}
        loss_func = loss(**kwargs)
        if reset_optimizer or self.use_optimizer is None:
            self.use_optimizer = optim(self.net.parameters(), lr=lr, **optim_kwargs)
            if lr_stepping:
                if lr_patience < 1:
                    lr_patience = int(lr_patience * num_epochs)
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.use_optimizer,
                    patience=lr_patience,
                )
        loss_record = np.zeros((num_epochs, num_steps))
        if val_set is not None:
            X_val, y_val = val_set
            X_val = list(self._setup_outsider(xi) for xi in X_val)
            y_val = self._setup_outsider(y_val)
            val_loss_record = np.zeros_like(loss_record)
        for j in range(num_epochs):
            for i, (X_batch, y_batch) in enumerate(gen):
                self.net.train()
                self.use_optimizer.zero_grad()
                loss = self.get_loss(X_batch, y_batch, loss_func, **forward_kwargs)
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
                loss_record[j, i] = loss.item()
                self.net.eval()
                with torch.no_grad():
                    if val_set is not None:
                        val_loss_record[j, i] = self.get_loss(
                            X_val, y_val, loss_func, **forward_kwargs
                        )
                    self.run_trackers(trackers)
                if i >= num_steps - 1:
                    break
            if lr_stepping:
                lr_scheduler.step(loss.detach())

        out_dict = {}
        out_dict["trackers"] = self.get_tracker_data(trackers)
        out_dict["loss"] = loss_record
        if val_set is not None:
            out_dict["val_loss"] = val_loss_record
        return out_dict


class GenericTrainingLoop:
    def fit(
        self,
        X,
        y,
        batch_size=100,
        ragged=False,
        **kwargs,
    ):
        X = self._setup_outsider(X)
        y = self._setup_outsider(y)
        gen = batch_generator(X, y, ragged=ragged, batch_size=batch_size)
        return self.fit_generator(gen, **kwargs)

    def training_step(self, X_batch, y_batch, loss_func, l1_reg=0, l2_reg=0, **kwargs):
        self.net.train()
        self.use_optimizer.zero_grad()
        loss = self.get_loss(X_batch, y_batch, loss_func, **kwargs)
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
        return loss.item()

    def validation_step(self, X_val, y_val, loss_func, **kwargs):
        self.net.eval()
        return self.get_loss(X_val, y_val, loss_func, **kwargs)

    def setup_optimizer(self, optim, lr, lr_stepping=False, lr_patience=3, **kwargs):
        use_optimizer = optim(self.net.parameters(), lr=lr, **kwargs)
        if lr_stepping:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                use_optimizer,
                patience=lr_patience,
            )
        else:
            lr_scheduler = None
        return use_optimizer, lr_scheduler

    def fit_generator(
        self,
        gen,
        num_steps=None,
        optim=None,
        lr=1e-3,
        loss=None,
        trackers=None,
        reset_optimizer=False,
        l2_reg=0,
        l1_reg=0,
        num_epochs=10,
        lr_stepping=False,
        lr_patience=3,
        val_set=None,
        optim_kwargs=None,
        loss_kwargs=None,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = len(gen)
        if trackers is None:
            trackers = []
        if optim is None:
            optim = torch.optim.Adam
        if optim_kwargs is None:
            optim_kwargs = {}
        if loss_kwargs is None:
            loss_kwargs = {}
        if loss is None:
            loss = nn.MSELoss
        loss_func = loss(**loss_kwargs)
        if reset_optimizer or self.use_optimizer is None:
            self.use_optimizer, lr_scheduler = self.setup_optimizer(
                optim,
                lr,
                lr_stepping=lr_stepping,
                lr_patience=lr_patience,
                **optim_kwargs,
            )
        loss_record = np.zeros((num_epochs, num_steps))
        if val_set is not None:
            X_val, y_val = list(self._setup_outsider(x) for x in val_set)
            val_loss_record = np.zeros_like(loss_record)
        for j in range(num_epochs):
            for i, (X_batch, y_batch) in enumerate(gen):
                loss_record[j, i] = self.training_step(
                    X_batch,
                    y_batch,
                    loss_func,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    **kwargs,
                )
                with torch.no_grad():
                    if val_set is not None:
                        val_loss_record[j, i] = self.validation_step(
                            X_val, y_val, loss_func, **kwargs
                        )
                    self.run_trackers(trackers)
                if i >= num_steps - 1:
                    break
            if lr_stepping:
                lr_scheduler.step(loss_record[j].mean())

        out_dict = {}
        out_dict["trackers"] = self.get_tracker_data(trackers)
        out_dict["loss"] = loss_record
        if val_set is not None:
            out_dict["val_loss"] = val_loss_record
        return out_dict


class DatasetSampled(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.args = args
        self.n_samples = args[0].shape[0]

    def __getitem__(self, key):
        out = list(arg[key] for arg in self.args)
        return out

    def __len__(self):
        return self.n_samples


class MultimodalDatasetSampled(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.Xs, self.y = args
        self.n_samples = len(self.y)

    def __getitem__(self, key):
        out1 = list(xi[key] for xi in self.Xs)
        out2 = self.y[key]
        return (out1, out2)

    def __len__(self):
        return self.n_samples


def _ragged_collation(x):
    feats, targs = [], []
    for f_i, t_i in x:
        feats.append(f_i)
        targs.append(t_i)
    return (
        torch.nested.as_nested_tensor(feats, layout=torch.jagged),
        torch.nested.as_nested_tensor(targs, layout=torch.jagged),
    )


def batch_generator(
    *data,
    batch_size=None,
    ragged=False,
    sampler_class=DatasetSampled,
    shuffle=True,
    **kwargs,
):
    if batch_size is None:
        batch_size = data[0].shape[0]
    data = sampler_class(*data)
    if ragged:
        collate_fn = _ragged_collation
    else:
        collate_fn = None
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, **kwargs
    )
    return loader


def multimodal_batch_generator(*data, **kwargs):
    return batch_generator(*data, **kwargs, sampler_class=MultimodalDatasetSampled)
