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
            split=0.9,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super(TNGDataset, self).__init__()
        self.cache = {}
        self.filepath = filepath
        self.device = device
        if cache_data:
            self.load_data()
            self._len = self.cache["kappa"].shape[0]
            self.pixels = self.cache["kappa"].shape[1]
            self.pixel_scale = self.cache["pixel_scale"]

        else:
            with h5py.File(filepath, "r") as hf:
                self._len = hf["kappa"].shape[0]
                self.pixels = hf["kappa"].shape[1]
                self.pixel_scale = hf["pixel_scale"][0]  # assumes it's all the same
        x = torch.linspace(-1, 1, self.pixels, device=device) * self.pixel_scale * self.pixels
        x, y = torch.meshgrid(x, x)
        self.coordinates = torch.stack([torch.ravel(x), torch.ravel(y)], dim=1)

        indices = list(range(self._len))
        self.train_split = np.random.choice(indices, replace=False, size=int(split * self._len)).tolist()
        self.test_split = list(set(indices).difference(set(self.train_split)))

    def __getitem__(self, index):
        out = {}
        train_index = self.train_split[index % len(self.train_split)]
        test_index = self.test_split[index % len(self.test_split)]
        if self.cache != {}:
            train_kappa = torch.tensor(self.cache["kappa"][train_index].ravel(), device=self.device)
            train_psi = torch.tensor(self.cache["psi"][train_index].ravel(), device=self.device)
            train_alpha = torch.flatten(torch.tensor(self.cache["alpha"][train_index], device=self.device), 0, 1)
            test_kappa = torch.tensor(self.cache["kappa"][test_index].ravel(), device=self.device)
            test_psi = torch.tensor(self.cache["psi"][test_index].ravel(), device=self.device)
            test_alpha = torch.flatten(torch.tensor(self.cache["alpha"][test_index], device=self.device), 0, 1)
        else:
            with h5py.File(self.filepath, "r") as hf:
                train_kappa = torch.tensor(hf["kappa"][train_index].ravel(), device=self.device)
                train_psi = torch.tensor(hf["psi"][train_index].ravel(), device=self.device)
                train_alpha = torch.flatten(torch.tensor(hf["alpha"][train_index], device=self.device), 0, 1)
                test_kappa = torch.tensor(hf["kappa"][test_index].ravel(), device=self.device)
                test_psi = torch.tensor(hf["psi"][test_index].ravel(), device=self.device)
                test_alpha = torch.flatten(torch.tensor(hf["alpha"][test_index], device=self.device), 0, 1)
        out["train"] = [
            self.coordinates,
            {
                "image": train_psi,
                "gradient": train_alpha,
                "laplace": train_kappa
            }
        ]
        out["test"] = [
            self.coordinates,
            {
                "image": test_psi,
                "gradient": test_alpha,
                "laplace": test_kappa
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = TNGDataset("/home/alexandre/Desktop/Projects/Lensing-SIREN/data/hkappa188hst_TNG100_rau_spl.h5")
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        print(batch["train"][1]["laplace"].shape)
        break