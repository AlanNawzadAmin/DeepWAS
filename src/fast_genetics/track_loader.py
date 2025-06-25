import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.fast_genetics.mmap_tracks import create_chunked_mmap, extract_windows

from .chunkedBuffer import SingleIOChunkedBuffer
from .get_tracks import allele_coset, bw_extract_residue_level_values, load_bigwigs, pl_extract_residue_level_values


@torch.compile(fullgraph=True, mode="max-autotune")
def fused_normalize_clip_convert(x: torch.Tensor, mean: torch.Tensor, stdv: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x - mean) / stdv, min=-100, max=100).to(torch.float16)


class TrackDataset(Dataset):
    def __init__(
        self,
        data_path,
        window_size_data=100,
        print_=False,
        chrom=np.arange(22) + 1,
        tracks_include=["fantom", "encode", "phylo"],
        max_n_snps=1000000,
        sumstats_type="cov_SMOKING_STATUS",
        fix_sumstats_D=False,
        use_z=False,
        D_as_feature=False,
        **kwargs,
    ):
        self.data_path = Path(data_path)
        self.data_path_tracks = Path("data")
        dbnsfp_path = self.data_path / Path("tracks/dbNSFP/dbNSFP5.1a")

        self.fix_sumstats_D = fix_sumstats_D
        self.use_z = use_z
        self.max_n_snps = max_n_snps
        self.pop = "EUR"
        self.print = print_
        self.window_size = window_size_data  # w bp window size (half on each side of SNP)
        self.D_as_feature = D_as_feature

        # sumstats
        if sumstats_type != "all":
            sumstats_file = self.data_path / Path(f"ukbb_sumstats/UKBB_409K/{sumstats_type}.sumstats")
            self.sumstats = pl.read_csv(sumstats_file, separator="\t")
            self.sumstats = {chr_num: self.sumstats.filter(self.sumstats["CHR"] == chr_num) for chr_num in chrom}
        elif sumstats_type == "all":
            self.sumstats = {
                chr_num: str(self.data_path / Path(f"ukbb_sumstats/merged_z_scores/chr{chr_num}.arrow"))
                for chr_num in chrom
            }

        # Get tracks
        self.tracks_include = tracks_include
        _, self.track_names = load_bigwigs(self.data_path, include=self.tracks_include, print_=self.print)
        self.dbnsfp_data = {chr_num: dbnsfp_path / f"filtered_chr{chr_num}.parquet" for chr_num in chrom}
        self.dbnsfp_col_names_load = [
            "ESM1b_score",
            "GERP++_RS",
            "SIFT_score",
            "PROVEAN_score",
            "fathmm-XF_coding_score",
            "AlphaMissense_score",
        ]
        self.dbnsfp_col_names = self.dbnsfp_col_names_load + D_as_feature * ["std", "EAF", "LLD"]
        try:
            track_stats = pickle.load(open("data/tracks/track_stats_2682.pkl", "rb"))
            geno_idx = [track_stats["geno_tracks"].index(name) for name in self.track_names]
            anno_idx = [track_stats["anno_tracks"].index(name) for name in self.dbnsfp_col_names]
            self.geno_mean = torch.tensor(track_stats["geno_mean"][geno_idx])[None, :, None]
            self.geno_stdv = torch.tensor(track_stats["geno_stdv"][geno_idx])[None, :, None]
            self.anno_mean = torch.tensor(track_stats["anno_mean"][anno_idx])[None]
            self.anno_stdv = torch.tensor(track_stats["anno_stdv"][anno_idx])[None]
        except Exception:
            self.geno_mean = 0
            self.geno_stdv = 1
            self.anno_mean = 0
            self.anno_stdv = 1
            print("WARNING: found no track statistics -- training will be unstable.")

    def __len__(self):
        return len(self.genotype_files)

    def __getitem__(self, idx):
        chr_num, snp_pos, site_var_id, genotypes, sign = self.get_snp_loc(idx)

        tik = time.time()
        # Get summary statistics for this region, by pos and mut
        chr_stats = self._get_sum_stats(chr_num)
        summary_stats = pl_extract_residue_level_values(
            chr_stats,
            site_var_id,
            col_names=["Beta", "EAF", "INFO"],
            id_names=["SNP", "POS", "A1", "A2"],
            return_sign=True,
        )
        summary_stats["sign"] = 2 * (summary_stats["sign"] == sign) - 1
        D = torch.sqrt(2 * summary_stats["EAF"] * (1 - summary_stats["EAF"]))
        if self.print:
            print("loading sum stats:", time.time() - tik)
        if self.fix_sumstats_D:
            summary_stats["Beta"] = summary_stats["Beta"] * D[:, None]
        if self.use_z:
            summary_stats["Beta"] = summary_stats["z"] / np.sqrt(407527)

        # Get dbNSFP for this region, by pos and mut
        # There can be duplicates for overlapping exons!
        tik = time.time()
        anno_tracks = pl_extract_residue_level_values(self._get_annos(chr_num), site_var_id)
        if self.D_as_feature:
            anno_tracks["D"] = D
            anno_tracks["EAF"] = summary_stats["EAF"]
            anno_tracks["LLD"] = torch.sqrt(genotypes**2).sum(-1)
        if self.print:
            print("loading dbnsfp:", time.time() - tik)
            tik = time.time()
        anno_tracks = torch.stack(list(anno_tracks.values()), dim=0).T
        anno_tracks = (anno_tracks - self.anno_mean) / self.anno_stdv
        # anno_tracks = anno_tracks.to(torch.bfloat16)

        geno_tracks = self._get_genomic_tracks(snp_pos, chr_num)
        tik = time.time()
        geno_tracks = fused_normalize_clip_convert(geno_tracks, self.geno_mean, self.geno_stdv)
        if self.print:
            print("normalization time:", time.time() - tik)
        return {
            "genotypes": genotypes,
            "region_stats": summary_stats,
            "geno_tracks": geno_tracks,
            "anno_tracks": anno_tracks,
            "position": snp_pos,
        }

    def _get_genomic_tracks_bw(self, snp_pos, chr_num):
        tik = time.time()
        track_bigwigs, _ = load_bigwigs(self.data_path, include=self.tracks_include, print_=self.print)
        if self.print:
            print("loading bws:", time.time() - tik)
        tik = time.time()
        genomic_tracks = bw_extract_residue_level_values(
            snp_pos,
            chr_num,
            track_bigwigs,
            self.window_size,
            print_=self.print,
        )
        if self.print:
            print("loading tracks:", time.time() - tik)
        return genomic_tracks

    def _get_genomic_tracks(self, snp_pos, chr_num):
        return self._get_genomic_tracks_bw(snp_pos, chr_num)

    def _get_annos(self, chr_num):
        data = pl.read_parquet(
            self.dbnsfp_data[chr_num],
            columns=self.dbnsfp_col_names_load + ["rs_dbSNP", "pos(1-based)", "ref", "alt"],
        )
        return data

    def _get_sum_stats(self, chr_):
        data = self.sumstats[chr_]
        if isinstance(data, str):
            return pl.read_ipc(data)
        else:
            return data

    def _make_var_ids(self, pos, a1, a2):
        var_ids = [f"{p}:{allele_coset(a + ':' + d)}" for p, a, d in zip(pos, a1, a2)]
        return var_ids

    def get_snp_loc(self, idx):
        raise NotImplementedError


def hash_string(text):
    hex_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return int(hex_hash, 16)


class TrackDatasetHDF5(TrackDataset):
    def __init__(
        self,
        create_hdf5=True,
        lazy_loading=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.create_hdf5 = create_hdf5
        self.mmapped_files = {}  # Will hold our memory maps
        self.loaded_mmaps = {}
        self.lazy_loading = lazy_loading
        self._initialize_mappings()  # Set up initial mappings

    def _load_mmaps(self, chr_num, hdf5_fname):
        self.loaded_mmaps[chr_num] = True
        for track in range(len(self.track_names)):
            track_key = f"track_{track}"
            track_prefix = Path(hdf5_fname) / f"chr{chr_num}/track_{track}"
            mmap_path = f"{track_prefix}.npy"
            file_size_mb = os.path.getsize(mmap_path) / 1024 / 1024
            if file_size_mb < 10:  # Threshold in MB
                self.mmapped_files[chr_num][track_key] = np.load(mmap_path)
                if self.print:
                    print(f"Loaded track {track} directly: {file_size_mb:.2f}KB")
            else:
                chunk_size = 1_000_000
                self.mmapped_files[chr_num][track_key] = SingleIOChunkedBuffer(
                    mmap_path, chunk_size=chunk_size, max_cached_chunks=4
                )
                if self.print:
                    print(f"Memory-mapped track {track}: {file_size_mb:.2f}KB")
        print(
            f"Chr: {chr_num}, fraction mmap'ed: {np.mean([hasattr(track, 'chunk_size') for _, track in self.mmapped_files[chr_num].items()]):.2f}"
        )

    def _initialize_mappings(self):
        """Set up memory mappings for all chromosomes we'll need."""
        for chr_num in range(1, 23):  # Assuming human chromosomes 1-22
            hdf5_fname, exists = self._get_hdf5_fname(chr_num)
            if exists:
                self.mmapped_files[chr_num] = {}
                self.loaded_mmaps[chr_num] = False
                if not self.lazy_loading:
                    self._load_mmaps(chr_num, hdf5_fname)
            else:
                print("Missing:", hdf5_fname)
        print("Loader initialized.")

    def close(self):
        for chr_num in list(self.mmapped_files.keys()):
            # for track_key, mmap in list(self.mmapped_files[chr_num].items()):
            #     if hasattr(mmap, "file"):
            #         mmap.close()
            # if mmap is not None:
            #     mmap._mmap.close()
            self.loaded_mmaps[chr_num] = False
            self.mmapped_files[chr_num].clear()

    def __getstate__(self):
        """Return state for pickling - exclude memory mappings."""
        state = self.__dict__.copy()
        self.close()
        state["mmapped_files"] = {}
        return state

    def __setstate__(self, state):
        """Restore state after unpickling and reinitialize memory mappings."""
        self.__dict__.update(state)
        # Recreate memory mappings on the new worker
        self._initialize_mappings()

    def __del__(self):
        self.close()
        self.mmapped_files.clear()
        self.loaded_mmaps.clear()

    def _get_hdf5_fname(self, chr_num):
        track_hdf5_path = os.path.join(self.data_path_tracks, "tracks/mmap")
        fname = str(hash_string("_".join(self.track_names))) + f"_chr{chr_num}"
        hdf5_fname = os.path.join(track_hdf5_path, fname)
        exists = fname in os.listdir(track_hdf5_path) and "metadata.json" in os.listdir(
            os.path.join(track_hdf5_path, fname)
        )
        if self.print:
            print(hdf5_fname)
        return hdf5_fname, exists

    def _get_genomic_tracks(self, snp_pos, chr_num):
        hdf5_fname, exists = self._get_hdf5_fname(chr_num)
        if not exists and self.create_hdf5:
            track_bigwigs, _ = load_bigwigs(self.data_path, include=self.tracks_include)
            create_chunked_mmap(track_bigwigs, [int(chr_num)], hdf5_fname)
            exists = True
        if exists:
            if not self.loaded_mmaps[chr_num]:
                self.close()
                self._load_mmaps(chr_num, hdf5_fname)
            tik = time.time()
            genomic_tracks = extract_windows(
                snp_pos,
                chr_num,
                hdf5_fname,
                n_tracks=len(self.track_names),
                window_size=self.window_size,
                mmapped_files=self.mmapped_files[chr_num],
                print_=self.print,
            )
        else:
            return self._get_genomic_tracks_bw(snp_pos, chr_num)
        if self.print:
            print("loading tracks:", time.time() - tik)
        return torch.tensor(genomic_tracks)
