from torch.utils.data import Dataset
import h5py
import numpy as np


class TNGDataset(Dataset):
    """
    Input params:
        filepath: path to the h5 file (prepared by fits_to_h5.py)
        cache_data: If true, load all the data into RAM.
    """
    def __init__(self, filepath, cache_data=False, transform=None):
        self.cache = {}
        self.transform = transform
        self.filepath = filepath
        if cache_data:
            self.load_data()
            self._len = self.cache["psi"].shape[0]
        else:
            with h5py.File(filepath, "r") as hf:
                self._len = hf["psi"].shape[0]

    def __getitem__(self, index):
        if self.cache != {}:
            psi = self.cache["psi"][index]
            kappa = self.cache["kappa"][index]
            alpha = self.cache["alpha"][index]
            return psi, alpha, kappa
        else:
            with h5py.File(self.filepath, "r") as hf:
                psi = hf["psi"][index]
                kappa = hf["kappa"][index]
                alpha = hf["alpha"][index]
                return psi, alpha, kappa

    def __len__(self):
        return self._len

    def load_data(self):
        with h5py.File(self.filepath, "r") as hf:
            self.cache["psi"] = np.array(hf["psi"][:])
            self.cache["kappa"] = np.array(hf["kappa"][:])
            self.cache["alpha"] = np.aaray(hf["alpha"][:])
