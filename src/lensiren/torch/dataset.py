import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class TNGDataset(Dataset):
    """
    Input params:
        filepath: path to the h5 file (prepared by fits_to_h5.py)
        cache_data: If true, load all the data into RAM.
    """
    def __init__(
            self,
            filepath,
            cache_data=False,
            indices=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super(TNGDataset, self).__init__()
        self.cache = {}
        self.filepath = filepath
        self.device = device
        if cache_data:
            self.load_data()
            self._total_len = self.cache["kappa"].shape[0]
            self.pixels = self.cache["kappa"].shape[1]
            self.pixel_scale = self.cache["pixel_scale"]

        else:
            with h5py.File(filepath, "r") as hf:
                self._total_len = hf["kappa"].shape[0]
                self.pixels = hf["kappa"].shape[1]
                self.pixel_scale = hf["pixel_scale"][0]  # assumes it's all the same
        x = torch.linspace(-1, 1, self.pixels, device=device) * self.pixel_scale * self.pixels
        x, y = torch.meshgrid(x, x)
        self.coordinates = torch.stack([torch.ravel(x), torch.ravel(y)], dim=1).requires_grad_()

        if indices is None:
            self.indices = list(range(self._total_len))
            self._len = self._total_len
        else:
            self.indices = indices
            self._len = len(indices)

    def __getitem__(self, index):
        index = self.indices[index]
        if self.cache != {}:
            train_kappa = torch.tensor(self.cache["kappa"][index].ravel(), device=self.device).view(-1, 1)
            train_psi = torch.tensor(self.cache["psi"][index].ravel(), device=self.device).view(-1, 1)
            train_alpha = torch.flatten(torch.tensor(self.cache["alpha"][index], device=self.device), 0, 1)
        else:
            with h5py.File(self.filepath, "r") as hf:
                train_kappa = torch.tensor(hf["kappa"][index].ravel(), device=self.device).view(-1, 1)
                train_psi = torch.tensor(hf["psi"][index].ravel(), device=self.device).view(-1, 1)
                train_alpha = torch.flatten(torch.tensor(hf["alpha"][index], device=self.device), 0, 1)
        out = [
            self.coordinates,
            {
                "image": train_psi,
                "gradient": train_alpha,
                "laplace": train_kappa
            }
        ]
        return out

    def __len__(self):
        return self._len

    def load_data(self):
        with h5py.File(self.filepath, "r") as hf:
            self.cache["kappa"] = np.array(hf["kappa"][:])
            self.cache["psi"] = np.array(hf["psi"][:])
            self.cache["alpha"] = np.array(hf["alpha"][:])

