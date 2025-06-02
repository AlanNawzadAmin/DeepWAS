import numpy as np
import torch
from pyliftover import LiftOver
from torch import nn

from .simulatedUKBBDataset import SimulatedUKBBDataset

lo = LiftOver("hg19", "hg38")

LINKS = {
    "exp": nn.Identity(),
    "relu": lambda x: torch.log(torch.clamp(x, min=1e-5)),
    "softplus": lambda x: torch.log(torch.clamp(nn.functional.softplus(x), min=1e-5)),
}


class SimRandInitDataset(SimulatedUKBBDataset):
    def __init__(
        self,
        nn_model=None,
        include_D=True,
        N=458303,
        M=11904924,
        log_sigma=np.log(0.5),
        window_size_data=100,
        print_=False,
        data_path="/scratch/aa11803/fast_genetics/",
        chrom=np.arange(22) + 1,
        tracks_include=["fantom", "encode", "phylo"],
        max_n_snps=1000000,
        link="exp",
        temperature=1.0,
        **kwargs,
    ):
        super().__init__(
            include_D=include_D,
            N=N,
            M=M,
            log_sigma=log_sigma,
            window_size_data=window_size_data,
            print_=print_,
            data_path=data_path,
            chrom=chrom,
            tracks_include=tracks_include,
            max_n_snps=max_n_snps,
            **kwargs,
        )
        self.nn_model = nn_model
        self.track_stats_idx = 1
        self.link = link
        self.temperature = temperature

    def model(self, geno_tracks, anno_tracks):
        log_preds = self.nn_model(geno_tracks, anno_tracks).detach()
        log_preds = LINKS[self.link](log_preds / self.temperature)
        return log_preds
