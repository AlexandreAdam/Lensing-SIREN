from astropy.io import fits
import h5py
import os, glob
import numpy as np
from tqdm import tqdm


def main(args):
    files = glob.glob(os.path.join(os.getenv('LSIREN_PATH'), "data", args.dataset, "*.fits"))
    pixels = fits.open(files[0])["PRIMARY"].data.shape[0]
    with h5py.File(os.path.join(os.getenv('LSIREN_PATH'), "data", args.dataset + ".h5"), "w") as hf:
        hf.create_dataset(name="psi", shape=[len(files), pixels, pixels, 1], dtype=np.float32)
        hf.create_dataset(name="alpha", shape=[len(files), pixels, pixels, 2], dtype=np.float32)
        hf.create_dataset(name="kappa", shape=[len(files), pixels, pixels, 1], dtype=np.float32)
        hf.create_dataset(name="shear1", shape=[len(files), pixels, pixels, 1], dtype=np.float32)
        hf.create_dataset(name="shear2", shape=[len(files), pixels, pixels, 1], dtype=np.float32)
        hf.create_dataset(name="kappa_id",  shape=[len(files)], dtype='i8')
        for i, file in enumerate(tqdm(files)):
            data = fits.open(file)
            hf["kappa"][i] = data["PRIMARY"].data[..., np.newaxis].astype(np.float32)
            hf["alpha"][i] = data["Deflection angles"].data.astype(np.float32)
            hf["psi"][i] = data["Lensing potential"].data[..., np.newaxis].astype(np.float32)
            hf["shear1"][i] = data["Shear1"].data[..., np.newaxis].astype(np.float32)
            hf["shear2"][i] = data["Shear2"].data[..., np.newaxis].astype(np.float32)
            hf["kappa_id"][i] = data["PRIMARY"].header["SUBID"]


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset",        required=True,      help="Name of dataset in data folder.")
    args = parser.parse_args()
    main(args)
