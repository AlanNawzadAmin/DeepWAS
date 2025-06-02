import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.fast_genetics.ChunkSampler import ChunkedDataset
from src.fast_genetics.dummyGenomicDataset import DummyGenomicDataset
from src.fast_genetics.okgDataset import OKGDataset
from src.fast_genetics.simulatedClean import SimCleanDataset
from src.fast_genetics.simulatedRandInit import SimRandInitDataset
from src.fast_genetics.ukbbDataset import UKBBDataset
from src.models.nets import get_model_setup


def get_dataset(name):
    CLS = {
        "ukbb": (UKBBDataset, CollatorUKBB),
        "okg": (OKGDataset, CollatorOKG),
        "dummy": (DummyGenomicDataset, CollatorOKG),
        "simClean": (SimCleanDataset, DefaultCollatorUKBB),
        "simRandInit": (SimRandInitDataset, DefaultCollatorUKBB),
    }
    return CLS[name]


def normalize_genos(tildeX):
    n_snp, n_pop = tildeX.shape
    mafs = tildeX.mean(-1)
    non_zero = torch.logical_and(mafs != 0, mafs != 1)
    stds = np.sqrt(mafs * (1 - mafs))[non_zero]
    X = (tildeX - mafs[:, None])[non_zero] / stds[:, None]
    D = stds
    return X, D, non_zero


def update_mask(batch, mask):
    if "mask" in batch.keys():
        batch["mask"] = torch.logical_and(mask, batch["mask"])
    else:
        batch["mask"] = mask


class DefaultCollatorUKBB:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = batch[0]
        return batch


class CollatorUKBB:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = batch[0]
        R = batch["genotypes"]
        sign = batch["region_stats"]["sign"]
        beta = batch["region_stats"]["Beta"]
        mafs = batch["region_stats"]["EAF"]
        stds = np.sqrt(mafs * (1 - mafs))
        D = stds
        non_zeros = D != 0
        geno_tracks = batch["geno_tracks"]
        anno_tracks = batch["anno_tracks"]
        processed = {
            "R": R,
            "D": D,
            "beta": (beta * sign.view(*sign.shape, *(1,) * (beta.ndim - 1))),
            "geno_tracks": geno_tracks,
            "anno_tracks": anno_tracks,
            "position": batch["position"],
        }
        update_mask(processed, non_zeros)
        return processed


class CollatorOKG:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = batch[0]
        pos = batch["position"]
        X, D, non_zeros = normalize_genos(batch["genotypes"])
        beta = batch["region_stats"]["Beta"]
        sign = batch["region_stats"]["sign"]
        geno_tracks = batch["geno_tracks"]
        anno_tracks = torch.stack(list(batch["anno_tracks"].values()), dim=0).T
        ukbb_mafs = batch["region_stats"]["EAF"]
        ukbb_stds = np.sqrt(ukbb_mafs * (1 - ukbb_mafs))
        ukbb_D = ukbb_stds
        non_zeros = torch.logical_and(non_zeros, ukbb_D != 0)
        processed = {
            "X": X,
            "D": D,
            "beta": (beta * sign.view(*sign.shape, *(1,) * (beta.ndim - 1)))[non_zeros],
            "geno_tracks": geno_tracks[non_zeros],
            "anno_tracks": anno_tracks[non_zeros],
            "position": pos[non_zeros],
        }
        return processed


class CollatorBeta:
    def __init__(self, drop_nan_beta=True, beta_inds=[]):
        self.drop_nan_beta = drop_nan_beta
        self.beta_inds = beta_inds

    def __call__(self, batch):
        if self.beta_inds != []:
            if batch["beta"].dim() > 1:
                batch["beta"] = batch["beta"][:, self.beta_inds]
        if self.drop_nan_beta:
            not_nan = torch.einsum("a...->a", torch.isnan(batch["beta"])) == 0
            update_mask(batch, not_nan)
        return batch


class CollatorMAF:
    def __init__(self, maf_thresh=0.0001, use_R_tilde=False):
        self.maf_thresh = maf_thresh
        self.use_R_tilde = use_R_tilde

    def __call__(self, batch):
        keep = (batch["D"] ** 2) >= self.maf_thresh
        update_mask(batch, keep)
        if self.use_R_tilde:
            D = batch["D"]
            if "X" in batch:
                batch["X"] = batch["X"] * D[:, None]
            if "R" in batch:
                batch["R"] = batch["R"] * D[:, None] * D[None, :]
            batch["beta"] = batch["beta"] * D.view(*D.shape, *(1,) * (batch["beta"].ndim - 1))
        return batch


class CollatorApplyMask:
    def __init__(self, explicit=False):
        self.explicit = explicit

    def __call__(self, batch):
        if self.explicit:
            mask = batch["mask"] if "mask" in batch else np.s_[:]
            mask = torch.where(mask)[0]
            # tik = time.time()
            for key in batch.keys():
                if key not in ["R", "mask"]:
                    batch[key] = torch.index_select(batch[key], 0, mask)
                # print("collated", key, "in", time.time()-tik); tik = time.time()
            if "R" in batch:
                # batch['R'] = batch['R'][mask][:, mask]
                batch["R"] = torch.index_select(batch["R"], 0, mask)
                batch["R"] = torch.index_select(batch["R"], 1, mask)
                # print("collated R in", time.time()-tik); tik = time.time()
        return batch


class CollatorLDWindow:
    def __init__(self, ld_dist_thresh=0):
        self.ld_dist_thresh = ld_dist_thresh

    def __call__(self, batch):
        if "position" in batch:
            pos = batch["position"]
            ld_window = (pos <= pos.max() - self.ld_dist_thresh) & (pos >= pos.min() + self.ld_dist_thresh)
            batch["ld_window"] = ld_window
            if "mask" in batch and (len(batch["mask"]) == len(batch["position"])):
                batch["ld_window"] = torch.logical_and(batch["ld_window"], batch["mask"])
        else:
            if "mask" in batch:
                batch["ld_window"] = batch["mask"]
        return batch


class ComposedCollator:
    def __init__(self, collators):
        self.collators = collators

    def __call__(self, batch):
        # tik = time.time()
        for i, collator in enumerate(self.collators):
            batch = collator(batch)
        # print("collated in", time.time()-tik); tik = time.time()
        return batch


def get_dataloaders(cfg):
    data_cls, make_collator = get_dataset(cfg.name)
    train_collator = make_collator()
    test_collator = make_collator()
    beta_collator = CollatorBeta(cfg.drop_nan_beta, cfg.beta_inds)
    maf_collator = CollatorMAF(cfg.maf_threshold, cfg.use_R_tilde)
    train_ldwindow_collator = CollatorLDWindow(cfg.ld_dist_thresh)
    test_ldwindow_collator = CollatorLDWindow(
        cfg.test_ld_dist_thresh if hasattr(cfg, "test_ld_dist_thresh") else cfg.ld_dist_thresh
    )
    if len(cfg.train_chrom) == 0:
        chrom = [chrom for chrom in np.arange(22) + 1 if chrom not in cfg.test_chrom]
    else:
        chrom = cfg.train_chrom

    train_data = data_cls(
        data_path=cfg.data_path,
        window_size_data=cfg.window_size_data,
        tracks_include=cfg.tracks_include,
        chrom=chrom,
        max_n_snps=cfg.max_n_snps,
        sumstats_type=cfg.sumstats_type,
        fix_sumstats_D=cfg.fix_sumstats_D,
        use_z=cfg.use_z,
        # print_=True,
        **OmegaConf.to_container(cfg.other_args, resolve=True),
    )
    test_data = data_cls(
        data_path=cfg.data_path,
        window_size_data=cfg.window_size_data,
        tracks_include=cfg.tracks_include,
        chrom=cfg.test_chrom,
        max_n_snps=cfg.test_max_n_snps,
        sumstats_type=cfg.sumstats_type,
        fix_sumstats_D=cfg.fix_sumstats_D,
        use_z=cfg.use_z,
        **OmegaConf.to_container(cfg.other_args, resolve=True),
    )
    if cfg.name in ["simRandInit", "simClean"]:
        model_class, nn_params = get_model_setup(cfg.other_args, train_data.track_names, train_data.dbnsfp_col_names)
        nn_model = model_class(**nn_params)
        # nn_model.eval()
        train_data.nn_model = nn_model
        test_data.nn_model = nn_model
    geno_track_names, anno_track_names = train_data.track_names, train_data.dbnsfp_col_names

    train_chunked_dataset = ChunkedDataset(
        train_data,
        chunk_size=cfg.chunk_size,
        shuffle=True,
        collate_fn=ComposedCollator(
            [train_collator, beta_collator, maf_collator, CollatorApplyMask(), train_ldwindow_collator]
        ),
        locs=[train_data._parse_filename(os.path.basename(f)) for f in train_data.genotype_files],
    )
    test_chunked_dataset = ChunkedDataset(
        test_data,
        chunk_size=cfg.chunk_size,
        shuffle=True,
        collate_fn=ComposedCollator(
            [test_collator, beta_collator, maf_collator, CollatorApplyMask(), test_ldwindow_collator]
        ),
        locs=[train_data._parse_filename(os.path.basename(f)) for f in test_data.genotype_files],
    )
    train_dataloader = DataLoader(
        train_chunked_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=cfg.n_workers,
        collate_fn=DefaultCollatorUKBB(),
        # collate_fn=ComposedCollator([train_collator, beta_collator, maf_collator, CollatorApplyMask(), ldwindow_collator]),
        persistent_workers=True,
        in_order=False,
    )
    test_dataloader = DataLoader(
        test_chunked_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=max(1, cfg.n_workers // 2),
        collate_fn=DefaultCollatorUKBB(),
        # collate_fn=ComposedCollator([test_collator, beta_collator, maf_collator, CollatorApplyMask(), ldwindow_collator]),
        persistent_workers=True,
        in_order=False,
    )
    return train_dataloader, test_dataloader, geno_track_names, anno_track_names
