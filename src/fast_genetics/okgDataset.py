import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from .track_loader import TrackDataset


class OKGDataset(TrackDataset):
    def __init__(
        self,
        window_size_data=100,
        print_=False,
        data_path="/scratch/aa11803/fast_genetics/",
        chrom=np.arange(22) + 1,
        tracks_include=["fantom", "encode", "phylo"],
        max_n_snps=1000000,
        **kwargs,
    ):
        super().__init__(
            window_size_data=window_size_data,
            print_=print_,
            data_path=data_path,
            chrom=chrom,
            tracks_include=tracks_include,
            max_n_snps=max_n_snps,
            **kwargs,
        )
        data_path = Path(data_path)

        # Get snplist and genotype files
        genotype_dir = data_path / "sparse_ld_download/genos"
        snplist_dir = data_path / "sparse_ld_download/snplist"
        self.genotype_dir = genotype_dir
        self.snplist_dir = snplist_dir
        self.genotype_files = [
            path
            for path in sorted(glob.glob(os.path.join(genotype_dir, f"*_{self.pop}.parquet")))  # .genos")))
            if any(f"_chr{chr_num}_" in path for chr_num in chrom)
        ]

    def get_snp_loc(self, idx):
        tik = time.time()
        # Extract chromosome and position range from filename
        genotype_file = self.genotype_files[idx]
        filename = os.path.basename(genotype_file)
        chr_num, start_pos, end_pos = self._parse_filename(filename)

        # Load genotypes
        genotypes = self._load_genotypes(genotype_file)
        if self.print:
            print("loading genotypes:", time.time() - tik)

        # make window
        n_snps = len(genotypes)
        start_widow = torch.randint(0, max([1, n_snps - self.max_n_snps + 1]), (1,))
        window = slice(start_widow, start_widow + self.max_n_snps)
        genotypes = genotypes[window]

        tik = time.time()
        # Load SNP information
        snplist_file = self._find_matching_snplist(chr_num, start_pos, end_pos)
        snp_pos, site_ids, site_var_id, signs = self._load_snp_info(snplist_file)
        if self.print:
            print("loading snplist:", time.time() - tik)
        return chr_num, snp_pos[window], site_var_id[window], genotypes, signs[window]

    def _parse_filename(self, filename):
        parts = filename.split("_")
        chr_num = parts[2][3:]  # Remove 'chr' prefix
        start_pos, end_pos = map(int, parts[3:5])
        return int(chr_num), start_pos, end_pos

    def _load_genotypes(self, genotype_file):
        # Load the genotype file (assuming it's a numpy array of 0s and 1s)
        return torch.tensor(pl.read_parquet(genotype_file).to_numpy(), dtype=torch.float32)

    def _find_matching_snplist(self, chr_num, start_pos, end_pos):
        pattern = f"1kg_chr{chr_num}_{start_pos}_{end_pos}.snplist"
        matching_files = glob.glob(os.path.join(self.snplist_dir, pattern))
        if matching_files:
            return matching_files[0]
        else:
            raise FileNotFoundError(f"No matching snplist file found for {pattern}")

    def _load_snp_info(self, snplist_file):
        snp_info = pd.read_csv(snplist_file)
        pos = torch.tensor(snp_info["position"].values, dtype=torch.long)
        var_ids = self._make_var_ids(
            snp_info["site_ids"].values, snp_info["anc_alleles"].values, snp_info["deriv_alleles"].values
        )
        return (
            pos,
            snp_info["site_ids"].values,
            var_ids,
            torch.tensor(snp_info["anc_alleles"].values < snp_info["deriv_alleles"].values, dtype=torch.long),
        )
