import numpy as np
import torch
from torch.utils.data import Dataset


class SimCleanDataset(Dataset):
    def __init__(
        self,
        N,
        M,
        log_sigma,
        window_size_data,
        n_tracks_genno,
        n_tracks_anno,
        max_n_snps,
        dataset_size=250,
        nn_model=None,
        **kwargs,
    ):
        self.nn_model = nn_model
        self.N = torch.tensor(N).to(torch.float32)
        self.M = torch.tensor(M).to(torch.float32)
        self.log_sigma = torch.tensor(log_sigma).to(torch.float32)
        self.wz = window_size_data
        self.n_tracks_genno = n_tracks_genno
        self.n_tracks_anno = n_tracks_anno
        self.max_n_snps = max_n_snps
        self.dataset_size = dataset_size
        self.n_draws = 1
        self.log_target_F_mean = torch.log((self.N / self.M) * (torch.exp(-self.log_sigma) - 1))
        self.log_running_F_mean = -torch.log(self.M)

        self.track_names = [f"{idx}" for idx in range(n_tracks_genno)]
        self.dbnsfp_col_names = [f"{idx}" for idx in range(n_tracks_anno)]
        self.genotype_files = []

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        batch = self.collator(idx)
        log_F = self.basic_predict(batch["geno_tracks"], batch["anno_tracks"], batch["D"])
        # self.update_running_F_mean(log_F)
        # log_F = log_F + self.log_target_F_mean - self.log_running_F_mean

        sim_beta = self.simulate_sumstats(log_F, batch["R"], self.N, self.log_sigma, seed=int(idx))
        batch["beta"] = sim_beta
        batch["ld_window"] = torch.ones_like(sim_beta, dtype=torch.bool)
        batch["log_F"] = log_F
        return batch

    def collator(self, batch):
        R = torch.eye(self.max_n_snps)
        D = torch.ones(self.max_n_snps)
        genno_tracks = torch.randn(self.max_n_snps, self.n_tracks_genno, self.wz)
        anno_tracks = torch.randn(self.max_n_snps, self.n_tracks_anno)
        processed = {
            "R": R,
            "D": D,
            "geno_tracks": genno_tracks,
            "anno_tracks": anno_tracks,
        }
        return processed

    def basic_predict(self, geno_tracks, anno_tracks, D):
        log_preds = self.nn_model(geno_tracks, anno_tracks).detach()
        # log_preds = torch.clamp(log_preds, min=-30, max=30)
        # log_preds = log_preds - torch.log(self.M)
        # #  log_preds = log_preds + 2 * self.D_weight * (torch.log(D) - torch.log(self.D_mean))
        # tilde_noise = torch.clamp(self.log_sigma, max=1.5) - torch.log(self.N)
        # log_preds = log_preds - tilde_noise  # in units of noise
        return log_preds

    def update_running_F_mean(self, log_F):
        n_snps = len(log_F)
        log_sum_F = torch.logsumexp(log_F, 0)
        cons = np.log(self.n_draws + n_snps)
        out = [log_sum_F - cons, self.log_running_F_mean + np.log(self.n_draws) - cons]
        out = torch.tensor(out, dtype=log_F.dtype, device=log_F.device)
        self.log_running_F_mean = torch.logsumexp(out, 0)
        self.n_draws += n_snps

    def simulate_sumstats(self, log_F, R, N, log_sigma, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        betas = torch.randn(self.max_n_snps, generator=g) * torch.exp(0.5 * log_F)
        hat_betas = betas
        # means = R @ betas
        # tilde_std = torch.exp(0.5 * log_sigma) / torch.sqrt(N)
        # hat_betas = tilde_std * (means + torch.randn(self.max_n_snps, generator=g))
        return hat_betas
