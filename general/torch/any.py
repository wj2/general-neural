
import torch
import torch.nn as nn


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


class DatasetSampled(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.args = args
        self.n_samples = args[0].shape[0]

    def __getitem__(self, key):
        out = list(arg[key] for arg in self.args)
        return out

    def __len__(self):
        return self.n_samples


def batch_generator(*data, batch_size=None, **kwargs):
    if batch_size is None:
        batch_size = data[0].shape[0]
    data = DatasetSampled(data)
    loader = torch.utils.data.DataLoader(batch_size=batch_size, **kwargs)
    return loader
