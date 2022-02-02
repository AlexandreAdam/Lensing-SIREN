from torchmeta.utils.data import MetaDataset
import h5py
import numpy as np


class KappaTNGDataset(MetaDataset):
    """
    Input params:
        filepath: path to the h5 file (prepared by fits_to_h5.py)
        cache_data: If true, load all the data into RAM.
    """
    def __init__(
            self,
            filepath,
            cache_data=False,
            transform=None,
            meta_train=False,
            meta_val=False,
            meta_test=False,
            meta_split=None
    ):
        super(KappaTNGDataset, self).__init__(meta_train, meta_val, meta_test, meta_split)
        self.cache = {}
        self.transform = transform
        self.filepath = filepath
        if cache_data:
            self.load_data()
            self._len = self.cache["kappa"].shape[0]
            self._size = self.cache["kappa"].shape[1]

        else:
            with h5py.File(filepath, "r") as hf:
                self._len = hf["kappa"].shape[0]
                self._size = hf["kappa"].shape[1]

    def __getitem__(self, index):
        if self.cache != {}:
            kappa = self.cache["kappa"][index]
            return kappa, kappa
        else:
            with h5py.File(self.filepath, "r") as hf:
                kappa = hf["kappa"][index]
                return kappa, kappa

    def __len__(self):
        return self._len

    def load_data(self):
        with h5py.File(self.filepath, "r") as hf:
            self.cache["kappa"] = np.array(hf["kappa"][:])

if __name__ == '__main__':
    dataset = KappaTNGDataset("/home/alexandre/Desktop/Projects/Lensing-SIREN/data/hkappa188hst_TNG100_rau_spl.h5", meta_train=True)
    for batch in dataset:
        print(batch)
        break