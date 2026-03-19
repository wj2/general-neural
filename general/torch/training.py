import numpy as np

import torch
import torch.nn as nn
import neurogym as ngym


def make_model_for_task(model_type, task, *args, **kwargs):
    n_inp = task.obs_dims
    n_out = task.action_dims
    model = model_type(n_inp, *args, n_out, **kwargs)
    return model


def make_epsilon_loss(epsilon=1, lam=1, **kwargs):
    mse = nn.MSELoss()

    def loss(x, y):
        n_dims = y.shape[-1]
        x_main = x[..., :n_dims]
        x_body = x[..., n_dims:]
        loss_main = mse(x_main, y)
        mask = torch.sum(x_body**2, axis=-1) >= epsilon**2
        mask = torch.unsqueeze(mask, -1)
        loss_body = torch.mean((x_body * mask) ** 2)
        return loss_main + lam * loss_body

    return loss


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_idx = self.X[idx]
        y_idx = self.y[idx]
        return X_idx, y_idx


def make_xy_dataloader(X, y, batch_size=16, **kwargs):
    dataset = XYDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)
    return dataloader


def train_model_on_data(model, X, y, val_dataset=None, batch_size=16, **kwargs):
    X = np.swapaxes(X, 1, 2)
    if val_dataset is not None:
        val_dataset = (np.swapaxes(val_dataset[0], 1, 2), val_dataset[1])
    dataset = make_xy_dataloader(X, y, batch_size=batch_size)
    return _train_model_epochs(model, dataset, val_dataset=val_dataset, **kwargs)


def _train_model_epochs(
    model,
    dataset,
    val_dataset=None,
    lr=1e-3,
    loss_func=None,
    print_interval=200,
    num_epochs=200,
    verbose=True,
    **kwargs,
):
    if loss_func is None:
        loss_func = nn.MSELoss

    device = model.device
    criterion = loss_func(**kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if val_dataset is not None:
        val_inputs, val_labels = val_dataset
        val_inputs = torch.from_numpy(val_inputs).type(torch.float).to(device)
        val_labels = torch.from_numpy(val_labels).type(torch.float).to(device)
        val_dataset = (val_inputs, val_labels)

    running_loss = 0.0
    losses = []
    val_losses = []
    for i in range(num_epochs):
        for inputs, labels in dataset:
            inputs = inputs.type(torch.float).to(device)
            labels = labels.type(torch.float).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())
            running_loss += loss.item()
        # print statistics
        if verbose:
            print("{:d} loss: {:0.5f}".format(i + 1, running_loss))
        running_loss = 0.0
        if val_dataset is not None:
            inputs_val, labels_val = val_dataset
            with torch.no_grad():
                pred_val, _ = model(inputs_val)
                val_loss = criterion(pred_val, labels_val)
                val_losses.append(val_loss)
                if verbose:
                    print("{:d} validation loss: {:0.5f}".format(i + 1, val_loss))

    out_dict = {
        "dataset": dataset,
        "loss": np.array(losses),
    }
    if val_dataset is not None:
        out_dict["pred_val"] = pred_val.detach().numpy(),
        out_dict["targ_val"] = labels_val.detach().numpy(),
        out_dict["loss_val"] = np.array(val_losses)
        out_dict["dataset_val"] = val_dataset
    return out_dict



def train_model_on_task(
    model,
    task,
    batch_size=16,
    seq_len=None,
    **kwargs,
):
    dataset = ngym.Dataset(task, batch_size=batch_size, seq_len=seq_len)
    return _train_model(model, dataset, **kwargs)


def _train_model(
    model,
    dataset,
    val_dataset=None,
    lr=1e-3,
    num_steps=2000,
    loss_func=None,
    print_interval=200,
    num_epochs=10,
    **kwargs,
):
    if loss_func is None:
        loss_func = nn.MSELoss

    device = model.device
    criterion = loss_func(**kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    losses = []
    val_losses = []
    for i in range(num_steps):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels).type(torch.float).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach())
        if val_dataset is not None:
            with torch.no_grad():
                inputs_val, labels_val = val_dataset()
                pred_val = model(inputs_val)
                val_loss = criterion(pred_val, labels_val)
                val_losses.append(val_loss)

        # print statistics
        running_loss += loss.item()
        if i % print_interval == print_interval - 1:
            print("{:d} loss: {:0.5f}".format(i + 1, running_loss / 200))
            running_loss = 0.0
    out_dict = {
        "dataset": dataset,
        "loss": losses,
    }
    if val_dataset is not None:
        out_dict["loss_val"] = val_losses
        out_dict["dataset_val"] = val_dataset
    return out_dict
