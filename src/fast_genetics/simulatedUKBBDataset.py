import numpy as np
import torch
from pyliftover import LiftOver

from src.fast_genetics.utils.train_fns import make_psd

from .ukbbDataset import UKBBDataset

lo = LiftOver("hg19", "hg38")


class SimulatedUKBBDataset(UKBBDataset):
    def __init__(
        self,
        data_path,
        include_D=True,
        N=458303,
        M=11904924,
        log_sigma=np.log(0.5),
        window_size_data=100,
        print_=False,
        chrom=np.arange(22) + 1,
        tracks_include=["fantom", "encode", "phylo"],
        max_n_snps=1000000,
        R_identity=False,
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
        self.R_identity = R_identity
        self.include_D = include_D
        self.N = torch.tensor(N).to(torch.float32)
        self.M = torch.tensor(M).to(torch.float32)
        self.log_sigma = torch.tensor(log_sigma).to(torch.float32)
        self.n_draws = 0.1
        self.log_target_F_mean = torch.log((self.N / self.M) * (torch.exp(-self.log_sigma) - 1))
        self.log_running_F_mean = -torch.log(self.M)

    def __getitem__(self, idx):
        batch = self.collator(super().__getitem__(idx))
        log_F = self.basic_predict(batch["geno_tracks"], batch["anno_tracks"], batch["D"])
        sim_beta = self.simulate_sumstats(log_F, batch["R"], self.N, self.log_sigma, seed=int(idx))
        batch["beta"], batch["log_F"] = sim_beta, log_F
        return batch

    def collator(self, batch):
        # gets rid of zeros
        R = batch["genotypes"]
        mafs = batch["region_stats"]["EAF"]
        stds = np.sqrt(mafs * (1 - mafs))
        D = stds
        non_zeros = torch.logical_and(D != 0, D != 1)
        geno_tracks = batch["geno_tracks"]
        anno_tracks = batch["anno_tracks"]
        processed = {
            "R": R[non_zeros][:, non_zeros],
            "D": D[non_zeros],
            "beta": batch["region_stats"]["Beta"][non_zeros],
            "geno_tracks": geno_tracks[non_zeros],
            "anno_tracks": anno_tracks[non_zeros],
            "position": batch["position"][non_zeros],
        }
        if self.R_identity:
            processed["R"] = torch.eye(len(processed["R"]), dtype=processed["R"].dtype, device=processed["R"].device)
        return processed

    def basic_predict(self, geno_tracks, anno_tracks, D):
        log_preds = self.model(geno_tracks.float(), anno_tracks) - torch.log(self.M)
        if self.include_D:
            log_preds = log_preds + 2 * torch.log(D + 1e-7)
        tilde_noise = torch.clamp(self.log_sigma, max=0.1) - torch.log(self.N)
        log_preds = log_preds - tilde_noise  # in units of noise
        return log_preds

    def model(geno_tracks, anno_tracks):
        raise NotImplementedError

    def update_running_F_mean(self, log_F):
        n_snps = len(log_F)
        log_sum_F = torch.logsumexp(log_F, 0)
        cons = np.log(self.n_draws + n_snps)
        out = [log_sum_F - cons, self.log_running_F_mean + np.log(self.n_draws) - cons]
        out = torch.tensor(out, dtype=log_F.dtype, device=log_F.device)
        self.log_running_F_mean = torch.logsumexp(out, 0)
        self.n_draws += n_snps

    def simulate_sumstats(self, log_F, R, N, log_sigma=0, seed=0, eps=1e-4):
        """Fix seed so we don't have to save the numbers for future epochs"""
        g = torch.Generator()
        g.manual_seed(seed)
        n_snps = len(log_F)
        betas = torch.randn(n_snps, generator=g) * torch.exp(0.5 * log_F)
        means = R @ betas
        cov = torch.linalg.cholesky(make_psd(R) + eps * torch.eye(len(R)))
        tilde_std = torch.exp(0.5 * log_sigma) / torch.sqrt(N)
        hat_betas = tilde_std * (means + cov @ torch.randn(n_snps, generator=g))
        return hat_betas
