import time

import torch

from .trainer import WASP


class LDSRTrainer(WASP):
    def __init__(
        self,
        model_class,
        n_pop,
        n_snp,
        herit=0.5,
        include_D=True,
        eps=1e-3,
        lr=1e-3,
        lr_sigma=2e-4,
        grad_clip_val=1,
        weight_decay=0,
        trace_estimate_n=10,
        seed=0,
        geno_track_names=[],
        anno_track_names=[],
        nn_params={},
        **kwargs,
    ):
        super().__init__(
            model_class,
            n_pop,
            n_snp,
            include_D=include_D,
            eps=eps,
            lr=lr,
            lr_sigma=lr_sigma,
            grad_clip_val=grad_clip_val,
            weight_decay=weight_decay,
            trace_estimate_n=trace_estimate_n,
            seed=seed,
            geno_track_names=geno_track_names,
            anno_track_names=anno_track_names,
            nn_params=nn_params,
            **kwargs,
        )
        self.save_hyperparameters(ignore=[])
        self.herit = herit

    def ldsr_forward(self, betas, R, log_F_N, D, ld_window):
        n_snp = len(log_F_N)
        dense_R = R.to_dense() + self.eps * torch.eye(n_snp, dtype=log_F_N.dtype, device=log_F_N.device)
        ld_mat = dense_R**2
        ld_scores = ld_mat.sum(0)[ld_window]
        diag_R = torch.diag(dense_R)[ld_window]
        chi_preds = (ld_mat @ torch.exp(log_F_N))[ld_window] + torch.exp(self.log_noise) * diag_R
        chi = (torch.sqrt(self.N) * betas[ld_window]) ** 2
        errs = chi_preds - chi
        errs_naive = torch.exp(self.log_noise) * diag_R - chi
        stds = self.N * ld_scores * self.herit / self.M + diag_R
        # get norm_ld scores
        # norm_R = dense_R / (torch.sqrt(diag_R[:, None] * diag_R[None, :]))
        # norm_ld_scores = (norm_R**2).sum(0)
        loss = ((errs / stds) ** 2) / ld_scores
        loss = loss.mean()
        loss_naive = ((errs_naive / stds) ** 2) / ld_scores
        loss_naive = loss_naive.mean()

        ### logging
        with torch.no_grad():
            log_F_beta = log_F_N - torch.log(self.N) + torch.log(self.M)
            F_beta = torch.exp(log_F_beta)

        diag = 1 / F_beta
        print(f"{tuple(R.shape)=} | Fraction in LD window: {ld_window.float().mean().item():1.3e}")
        print(f"diag max:     {1 / diag.min().item():1.3e} | diag min:     {1 / diag.max().item():1.3e}")
        print(f"inv_diag max: {diag.max().item():1.3e} | inv_diag min: {diag.min().item():1.3e}")
        print(
            "size of chi:",
            chi.mean().item(),
            "size of stds:",
            stds.mean().item(),
            "size of chi preds:",
            chi_preds.mean().item(),
            "size of errors:",
            torch.abs(errs / stds).mean().item(),
            "max chi:",
            chi.max().item(),
        )
        info = {
            "loss": loss,
            "delta_loss": loss - loss_naive,
            "err": errs.mean() / n_snp,
            "mean_log_F": log_F_beta.mean().item(),
            "mean_F": F_beta.mean().item(),
        }
        return loss, info

    def take_step(self, batch, return_val=False):
        print(f"Current step: {self.global_step}")
        tik = time.time()
        R = self._get_R(batch)  # [M, M]
        beta = batch["beta"]  # [M]
        R, beta = self.make_psd(R, beta)
        D = batch["D"]  # [M]
        ld_window = batch["ld_window"]
        geno_tracks = batch["geno_tracks"].float()  # [M, F_1, window_size]
        anno_tracks = batch["anno_tracks"]  # [M, F_2]
        print("time to load:", time.time() - tik)
        tik = time.time()
        log_F = self.basic_predict(geno_tracks, anno_tracks, D)
        torch.cuda.synchronize()
        print("time to log f:", time.time() - tik)
        tik = time.time()
        tik = time.time()
        loss, info = self.ldsr_forward(beta, R, log_F, D, ld_window)
        if "log_F" in batch:
            self.compare_against_truth(batch, log_F, info)

        if return_val:
            betas, R, log_F, sigma = self.change_precision(beta, R, log_F)
            loss_val, info_val = self(beta, R, log_F, D, ld_window, sigma)
            with torch.no_grad():
                loss_no_F, info_no_F = self(beta, R, -1000000 + 0 * log_F, D, ld_window, sigma)
                info_val["delta_loss"] = info_val["loss"] - info_no_F["loss"]
                info_val["delta_err"] = info_val["err"] - info_no_F["err"]
                info_val["delta_logdet"] = info_val["logdet"] - info_no_F["logdet"]
        else:
            loss_val = 0
            info_val = 0
        print("time to calc loss:", time.time() - tik)
        tik = time.time()
        return (
            loss,
            info,
        ) + return_val * (
            loss_val,
            info_val,
        )

    def validation_step(self, batch, batch_idx):
        if (batch["geno_tracks"].shape[0] <= 1) or ("ld_window" in batch and batch["ld_window"].sum() == 0):
            return None
        loss, info, loss_val, info_val = self.take_step(batch, return_val=True)

        print(
            "*=" * 50
            + f"\nTest LDSR Loss: {loss.item():1.3e} | Test Loss: {loss_val.item():1.3e} | log_F: {info['mean_log_F']:1.3e} | log_sig: {self.log_noise.item():1.3e}\n"
            + "*=" * 50
        )
        for key, val in info.items():
            self.log(f"val_ldsr_{key}", val, on_step=False, on_epoch=True, sync_dist=True)
        for key, val in info_val.items():
            self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True)
        if "log_F" in batch:
            for key, val in info.items():
                if "error" in key:
                    self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True)
        return info
