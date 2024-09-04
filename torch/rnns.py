import torch
import torch.nn as nn


def sample_model_responses(task, model, n_samples=1000):
    info, inputs, targets = task.sample_trials(n_samples)
    out = []
    for i, inp in enumerate(inputs):
        inp_use = torch.from_numpy(inp).type(torch.float).to(model.device)
        out.append(model.forward(inp_use))
    return info, inputs, targets, out


class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity

    Notes:
        taken from https://github.com/gyyang/nn-brain/blob/master/RNN_tutorial.ipynb
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        tau=100,
        dt=None,
        transfer_function=torch.relu,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.transfer_function = transfer_function
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = self.transfer_function(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        shape = input.shape[:-1]
        output = torch.zeros(shape + (self.hidden_size,)).to(input.device)
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output[i] = hidden

        return output, hidden


class EmbodiedCTRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        body_size,
        body_dynamics=None,
        tau=100,
        dt=None,
        transfer_function=torch.relu,
        dynamics_sigma=0,
        **kwargs,
    ):
        super().__init__()
        if body_dynamics is None:
            body_dynamics = torch.arange(body_size, requires_grad=False)
        else:
            body_dynamics = torch.tensor(body_dynamics, requires_grad=False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.body_size = body_size
        self.tau = tau
        self.transfer_function = transfer_function
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2b = nn.Linear(hidden_size, body_size)
        self.b2h = nn.Linear(body_size, hidden_size)
        self.body_mask = torch.zeros(body_size, requires_grad=False)
        self.body_mask[body_dynamics] = 1

    def init_hidden(self, input_shape):
        shape = input_shape[1:-1]
        return torch.zeros(shape + (self.hidden_size,))

    def init_body(self, input_shape):
        shape = input_shape[1:-1]
        return torch.zeros(shape + (self.body_size,))

    def recurrence(self, input, hidden, body, dyn_sigma=None):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        
        h_new = self.transfer_function(
            self.input2h(input) + self.h2h(hidden) + self.b2h(body)
        )
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha

        shape = (1,) * (len(body.shape) - 1) + (-1,)
        bm = torch.reshape(self.body_mask, shape)
        b_new = body * bm + self.alpha * self.h2b(hidden)
        return h_new, b_new

    def forward(self, input, hidden=None, body=None, **kwargs):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        if body is None:
            body = self.init_body(input.shape).to(input.device)

        # Loop through time
        shape = input.shape[:-1]
        hidden_output = torch.zeros(shape + (self.hidden_size,)).to(
            input.device
        )
        body_output = torch.zeros(shape + (self.body_size,)).to(input.device)
        steps = range(input.size(0))
        for i in steps:
            hidden, body = self.recurrence(input[i], hidden, body, **kwargs)
            hidden_output[i] = hidden
            body_output[i] = body

        return hidden_output, body_output, hidden


class EmbodiedRecurrent(nn.Module):
    def __init__(self, inp_size, num_h, out_size, net_type=EmbodiedCTRNN, **kwargs):
        super(EmbodiedRecurrent, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hidden_dim = num_h
        self.out_dim = out_size
        self.in_dim = inp_size
        self.emb_net = net_type(inp_size, num_h, out_size, device=self.device, **kwargs)

    def forward(self, x, hidden=None, body=None):
        rnn_out, body_out, _ = self.emb_net(x, hidden=hidden, body=body)
        return body_out, rnn_out


class SimpleRecurrent(nn.Module):
    def __init__(self, inp_size, num_h, out_size, net_type=nn.RNN):
        super(SimpleRecurrent, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hidden_dim = num_h
        self.out_dim = out_size
        self.in_dim = inp_size
        self.recurrent_net = net_type(inp_size, num_h, device=self.device)
        self.linear = nn.Linear(num_h, out_size, device=self.device)

    def forward(self, x, hidden=None):
        rnn_out, _ = self.recurrent_net(x, hidden=hidden)
        x = self.linear(rnn_out)
        return x, rnn_out


class SimpleCTRNN(SimpleRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleCTRNN, self).__init__(*args, **kwargs, net_type=CTRNN)


class SimpleRNN(SimpleRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleRNN, self).__init__(*args, **kwargs, net_type=nn.RNN)


class SimpleLSTM(SimpleRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleLSTM, self).__init__(*args, **kwargs, net_type=nn.LSTM)


class SimpleGRU(SimpleRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleGRU, self).__init__(*args, **kwargs, net_type=nn.GRU)
