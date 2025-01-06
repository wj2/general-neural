import torch


def lp_activity_reg(net, stim, reg_layers, p=2):
    activation = 0
    x = stim
    for i, layer in enumerate(net):
        x = layer(x)
        if reg_layers[i]:
            activation += torch.sum(torch.mean(torch.abs(x) ** p, axis=0))
    return activation / (i + 1)


def l2_activity_reg(net, stim, reg_layers):
    return lp_activity_reg(net, stim, reg_layers, p=2)


def l1_activity_reg(net, stim, reg_layers):
    return lp_activity_reg(net, stim, reg_layers, p=1)
