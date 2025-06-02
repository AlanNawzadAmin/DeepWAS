import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sparse
import torch
from pyliftover import LiftOver

from src.fast_genetics.utils.train_fns import make_psd

from .get_tracks import pl_extract_residue_level_values
from .track_loader import TrackDatasetHDF5

lo = LiftOver("hg19", "hg38")


class UKBBDataset(TrackDatasetHDF5):
    def __init__(
        self,
        data_path="/scratch/aa11803/data/",
        window_size_data=100,
        print_=False,
        chrom=np.arange(22) + 1,
        tracks_include=["fantom", "encode", "phylo"],
        max_n_snps=1000000,
        processed_gen_dir="dense_ld_mats_psd_t0",
        psd_eval_thresh=0.0,
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
        genotype_dir = data_path / "ukbb_windows/ld_mats"
        snplist_dir = data_path / "ukbb_windows/snplists"
        self.chrom = chrom
        self.genotype_dir = genotype_dir
        self.snplist_dir = snplist_dir
        self.genotype_files = [
            [path for path in sorted(glob.glob(os.path.join(genotype_dir, "*.npz"))) if f"chr{chr_num}_" in path]
            for chr_num in chrom
        ]
        geno_starts = [
            [self._parse_filename(os.path.basename(path))[1] for path in chr_paths] for chr_paths in self.genotype_files
        ]
        self.genotype_files = [
            np.array(chr_paths)[np.argsort(starts)] for chr_paths, starts in zip(self.genotype_files, geno_starts)
        ]
        self.genotype_files = [path for chr_paths in self.genotype_files for path in chr_paths]
        self.processed_gen_dir = processed_gen_dir
        self.psd_eval_thresh = psd_eval_thresh

    def get_snp_loc(self, idx):
        tik = time.time()
        # Extract chromosome and position range from filename
        genotype_file = self.genotype_files[idx]
        filename = os.path.basename(genotype_file)
        chr_num, start_pos, end_pos = self._parse_filename(filename)

        tik = time.time()
        # Load SNP information
        snplist_file = self._find_matching_snplist(chr_num, start_pos, end_pos)
        snp_pos, site_ids, site_var_id, sign = self._load_snp_info(snplist_file)
        if self.print:
            print("loading snplist:", time.time() - tik)

        # filter for betas
        chr_stats = self._get_sum_stats(chr_num)
        summary_stats = pl_extract_residue_level_values(
            chr_stats,
            site_var_id,
            col_names=["Beta", "EAF", "INFO"],
            id_names=["SNP", "POS", "A1", "A2"],
            return_sign=True,
        )
        maf = summary_stats["EAF"]
        non_zero = (maf * (1 - maf)) > 0
        snp_pos, site_ids, site_var_id, sign = (
            snp_pos[non_zero],
            site_ids[non_zero],
            np.array(site_var_id)[non_zero],
            sign[non_zero],
        )

        # make window
        n_snps = len(snp_pos)
        start_widow = torch.randint(0, max([1, n_snps - self.max_n_snps + 1]), (1,))
        window = slice(start_widow, start_widow + self.max_n_snps)

        # Load genotypes
        genotypes = self._load_genotypes(genotype_file, window, non_zero)
        if self.print:
            print("loading genotypes:", time.time() - tik)

        return chr_num, snp_pos[window], site_var_id[window], genotypes, sign[window]

    def _parse_filename(self, filename):
        parts = filename.split(".")[0].split("_")
        chr_num = parts[0][3:]  # Remove 'chr' prefix
        start_pos, end_pos = map(int, parts[1:3])
        return int(chr_num), start_pos, end_pos

    def _load_genotypes(self, genotype_file, window, non_zero):
        # Load the genotype file (its a "sparse" matrix of the upper triangle with 0.5 down the diag)
        modified_genotype_file = genotype_file.replace(".npz", ".arrow").replace("ld_mats", self.processed_gen_dir)
        if os.path.exists(modified_genotype_file):
            with pa.memory_map(modified_genotype_file) as source:
                tensor = pa.ipc.read_tensor(source)
        else:
            print("loading", genotype_file)
            mat = sparse.load_npz(genotype_file)
            print("done loading")
            mat = mat.toarray()[non_zero][:, non_zero]
            print("matrix is", mat.shape)
            mat = mat + mat.T
            # mat = torch.tensor(mat, dtype=torch.float32)
            # torch.save(mat, modified_genotype_file)
            print("getting eigh")
            tensor = pa.Tensor.from_numpy(
                make_psd(torch.from_numpy(mat).to("cuda"), thresh=self.psd_eval_thresh)
                .cpu()
                .numpy()  # this is hopeless without cuda
            )  # fast, direct conversion
            print("done eigh")
            with pa.OSFile(modified_genotype_file, "wb") as f:
                pa.ipc.write_tensor(tensor, f)
        mat = torch.from_numpy(tensor.to_numpy()).to(torch.float32)
        mat = mat[window][:, window]
        return mat

    def _find_matching_snplist(self, chr_num, start_pos, end_pos):
        pattern = f"chr{chr_num}_{start_pos}_{end_pos}.gz"
        matching_files = glob.glob(os.path.join(self.snplist_dir, pattern))
        if matching_files:
            return matching_files[0]
        else:
            raise FileNotFoundError(f"No matching snplist file found for {pattern}")

    def _load_snp_info(self, snplist_file):
        snp_info = pd.read_table(snplist_file, sep="\\s+")
        pos = snp_info["position"].values
        chr_num = snp_info["chromosome"][0]
        pos_set = [lo.convert_coordinate(f"chr{chr_num}", i) for i in pos]
        pos = np.array([i[0][1] if (len(i) > 0 and i[0][0] == f"chr{chr_num}") else 100000 for i in pos_set]).astype(
            int
        )
        if self.print:
            print("n found pos:", np.mean(pos != 100000))
        pos = torch.tensor(pos, dtype=torch.long)
        var_ids = self._make_var_ids(snp_info["rsid"].values, snp_info["allele1"].values, snp_info["allele2"].values)
        if self.print:
            print("frac w rsid:", np.mean(["rs" == id_[:2] for id_ in snp_info["rsid"].values]))
        return (
            pos,
            snp_info["rsid"].values,
            var_ids,
            torch.tensor(snp_info["allele1"].values < snp_info["allele2"].values, dtype=torch.long),
        )


class UKBBDatasetRaw(UKBBDataset):
    def _load_genotypes(self, genotype_file, window, non_zero):
        # Load the genotype file (its a "sparse" matrix of the upper triangle with 0.5 down the diag)
        mat = sparse.load_npz(genotype_file)
        mat = mat.toarray()[non_zero][:, non_zero]
        mat = mat + mat.T
        # mat = torch.tensor(mat, dtype=torch.float32)
        # torch.save(mat, modified_genotype_file)
        tensor = pa.Tensor.from_numpy(mat)  # fast, direct conversion
        mat = torch.from_numpy(tensor.to_numpy()).to(torch.float32)
        mat = mat[window][:, window]
        return mat
