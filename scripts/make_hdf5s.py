import argparse
import os

import numpy as np

from src.fast_genetics.ukbbDataset import UKBBDataset

parser = argparse.ArgumentParser()
parser.add_argument("r", type=int)
args = parser.parse_args()

r = args.r

tracks = [["phylo", "big_encode", "fantom"]]

data = UKBBDataset(tracks_include=tracks, sumstats_type="all", window_size=10, print_=True, create_hdf5=True)
chrs = [s.split("chr")[-1].split("_")[0] for s in data.genotype_files]
_, indxs = np.unique(chrs, return_index=True)
print(_, indxs)
try:
    data[indxs[r]]
except OSError:
    hdf5_fname = data._get_hdf5_fname(chrs[indxs[r]])
    os.remove(hdf5_fname)
    data[indxs[r]]
