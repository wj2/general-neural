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


def train_model_on_task(
    model,
    task,
    batch_size=16,
    seq_len=None,
    lr=1e-3,
    num_steps=2000,
    loss_func=None,
    print_interval=200,
    **kwargs,
):
    dataset = ngym.Dataset(task, batch_size=batch_size, seq_len=seq_len)
    if loss_func is None:
        loss_func = nn.MSELoss

    device = model.device
    criterion = loss_func(**kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
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

        # print statistics
        running_loss += loss.item()
        if i % print_interval == print_interval - 1:
            print("{:d} loss: {:0.5f}".format(i + 1, running_loss / 200))
            running_loss = 0.0
    return dataset
