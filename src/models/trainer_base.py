import time

import cola
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from transformers import get_linear_schedule_with_warmup


class SNPTrainer(pl.LightningModule):
    def __init__(
        self,
        model_class,
        n_pop,
        n_snp,
        geno_track_names,
        anno_track_names,
        include_D=True,
        eps=1e-3,
        lr=1e-3,
        lr_sigma=2e-4,
        grad_clip_val=1,
        weight_decay=0,
        trace_estimate_n=10,
        seed=0,
        nn_params={},
        beta_eval_thresh=0.0,
        start_log_sigma=0.0,
        train_sigma=True,
        wandb=True,
        train_D=False,
        normalize_tracks=False,
        max_batch_size=8192,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.N = torch.tensor(n_pop).to(torch.float32)
        self.M = torch.tensor(n_snp).to(torch.float32)
        self.beta_eval_thresh = beta_eval_thresh
        self.include_D = include_D
        self.lr = lr
        self.lr_sigma = lr_sigma
        self.batch_size_window = None
        self.grad_clip_val = grad_clip_val
        self.weight_decay = weight_decay
        self.eps = eps
        self.model = model_class(**nn_params)
        self.max_batch_size = max_batch_size
        log_noise = torch.tensor(start_log_sigma, dtype=torch.float32)
        if train_sigma:
            self.log_noise = nn.Parameter(log_noise)
        else:
            self.register_buffer("log_noise", log_noise)
        D_weight = torch.tensor(float(self.include_D))
        if train_D:
            self.D_weight = nn.Parameter(D_weight)
        else:
            self.register_buffer("D_weight", torch.tensor(float(self.include_D)))
        self.register_buffer("D_mean", torch.tensor(1.0))
        self.trace_estimate_n = trace_estimate_n
        self.wandb = wandb
        self.normalize_tracks = normalize_tracks

    def apply_model(self, norm_geno_tracks, norm_anno_tracks):
        if torch.cuda.is_available():
            return self.process_in_batches_cuda(norm_geno_tracks, norm_anno_tracks)
        else:
            return self.model(norm_geno_tracks, norm_anno_tracks)

    def process_in_batches_cuda(self, norm_geno_tracks, norm_anno_tracks, init_batch_size=128):
        total_samples = norm_geno_tracks.shape[0]
        all_log_preds = []
        batch_size = init_batch_size if self.batch_size_window is None else self.batch_size_window
        set_new_batch_size = (self.batch_size_window is None) and len(norm_geno_tracks) > init_batch_size

        torch.cuda.reset_peak_memory_stats()
        if set_new_batch_size:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_mem = torch.cuda.memory_allocated()
            memory_usages = []

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_geno = norm_geno_tracks[start_idx:end_idx]
            batch_anno = norm_anno_tracks[start_idx:end_idx]
            torch.cuda.empty_cache()
            print(start_idx, torch.cuda.max_memory_allocated())
            log_preds = self.model(batch_geno, batch_anno)
            all_log_preds.append(log_preds)
            if set_new_batch_size:
                memory_usages.append(torch.cuda.max_memory_allocated() - initial_mem)

        if set_new_batch_size:
            avg_memory = sum(memory_usages[:-1]) / len(memory_usages[:-1])
            available_mem = torch.cuda.get_device_properties(0).total_memory * 0.6
            self.batch_size_window = min(
                max(batch_size, int((available_mem / avg_memory) * batch_size)), self.max_batch_size
            )
            print("!!! Arrived at batch size !!!", self.batch_size_window)

        return torch.cat(all_log_preds, dim=0)

    def basic_predict(self, geno_tracks, anno_tracks, D):
        log_preds = self.apply_model(geno_tracks, anno_tracks)
        # log_preds = torch.clamp(log_preds, min=-30, max=30)
        # log_preds = log_preds - torch.log(self.M)
        # log_preds = log_preds + 2 * self.D_weight * (torch.log(D) - torch.log(self.D_mean))
        # tilde_noise = torch.clamp(self.log_noise, max=1.5) - torch.log(self.N)
        # log_preds = log_preds - tilde_noise  # in units of noise
        return log_preds

    def _get_R(self, batch):
        if "R" in batch:
            R = batch["R"]
            return cola.ops.Dense(R)  # [M, M]
        elif "X" in batch:
            X = batch["X"]  # [M, N_sample]
            X = cola.ops.Dense(X)
            return X @ X.T / X.shape[1]  # [M, M]

    def forward(self, betas, R, log_F_tilde, D, ld_window, sigma):
        pass

    def regularize_R(self, R):
        R = R + self.eps * cola.ops.I_like(R)
        return R

    def make_psd(self, R, beta):
        dtype = R.dtype
        B = R.to_dense()
        B = B.to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(B)
        eigvals = torch.maximum(eigvals, torch.tensor(0.0))
        R_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        R_psd = (R_psd + R_psd.T) / 2.0
        R_psd = R_psd.to(dtype)
        R_psd = cola.ops.Dense(R_psd)
        R_psd = self.regularize_R(R_psd)

        beta = beta.to(torch.float64)
        mask = (eigvals <= self.beta_eval_thresh)[:, None]
        beta = beta - (eigvecs @ (mask * eigvecs.T)) @ beta
        beta = beta.to(dtype)

        return R_psd, beta

    def take_step(self, batch):
        tik = time.time()
        print(f"Current step: {self.global_step}")
        R = self._get_R(batch)  # [M, M]
        betas = batch["beta"]  # [M]
        D = batch["D"]  # [M]
        ld_window = batch["ld_window"]
        geno_tracks = batch["geno_tracks"].float()  # [M, F_1, window_size]
        anno_tracks = batch["anno_tracks"].float()  # [M, F_2]
        print("time to load:", time.time() - tik)
        tik = time.time()

        log_F = self.basic_predict(geno_tracks, anno_tracks, D)
        torch.cuda.synchronize()
        print("time to log f:", time.time() - tik)
        tik = time.time()
        betas, R, log_F, sigma = self.change_precision(betas, R, log_F)
        loss, info = self(betas, R, log_F, D, ld_window, sigma)
        print("time to calculate loss:", time.time() - tik)
        tik = time.time()

        self.add_delta_loss(betas, R, log_F, D, ld_window, sigma, info)
        if "log_F" in batch:
            self.compare_against_truth(batch, log_F, info)
        print("rest of calcs:", time.time() - tik)
        tik = time.time()
        return loss, info

    def change_precision(self, beta, R, log_F):
        sigma = torch.exp(self.log_noise)
        if self.use_double:
            beta = beta.double()
            R = R.to_dense().double()
            log_F = log_F.double()
            sigma = torch.exp(self.log_noise).double()
        return beta, R, log_F, sigma

    def add_delta_loss(self, betas, R, log_F, D, ld_window, sigma, info):
        with torch.no_grad():
            loss_no_F, info_no_F = self(betas, R, -1e6 + 0 * log_F, D, ld_window, sigma)
            info["delta_loss"] = info["loss"] - info_no_F["loss"]
            info["delta_err"] = info["err"] - info_no_F["err"]
            info["delta_logdet"] = info["logdet"] - info_no_F["logdet"]

    def compare_against_truth(self, batch, log_F_tilde, info):
        # F = torch.exp(log_F_tilde + self.log_noise + torch.log(self.M / self.N))
        # F_truth = torch.exp(batch["log_F"] + self.log_noise + torch.log(self.M / self.N))
        F = torch.exp(log_F_tilde + torch.log(self.M / self.N))
        F_truth = torch.exp(batch["log_F"] + torch.log(self.M / self.N))
        info["F_error_mae"] = torch.mean(torch.abs(F - F_truth))

        info["log_F_tilde_error_mae"] = torch.mean(torch.abs(log_F_tilde - batch["log_F"]))
        info["log_F_tilde_error_rsme"] = torch.sqrt(torch.mean((log_F_tilde - batch["log_F"]) ** 2.0))
        F_tilde = torch.exp(log_F_tilde)
        F_tilde_truth = torch.exp(batch["log_F"])
        info["F_tilde_error_mae"] = torch.mean(torch.abs(F_tilde - F_tilde_truth))
        info["F_tilde_error_rsme"] = torch.sqrt(torch.mean((F_tilde - F_tilde_truth) ** 2.0))
        return info

    def training_step(self, batch, batch_idx):
        tik = time.time()
        if (batch["geno_tracks"].shape[0] <= 1) or ("ld_window" in batch and batch["ld_window"].sum() == 0):
            return None
        loss, info = self.take_step(batch)

        print(
            "*=" * 50
            + f"\nTrain Loss: {loss.item():1.3e} | log_F: {info['mean_log_F']:1.3e} | sig: {torch.exp(self.log_noise).item():1.3e}\n"
            + "*=" * 50
        )
        for key, val in info.items():
            self.log(f"train_{key}", val, sync_dist=True)
        self.log("log sigma", self.log_noise.item(), sync_dist=True)
        self.log("D weight", self.D_weight.item(), sync_dist=True)
        self.log("sigma", torch.exp(self.log_noise), sync_dist=True)
        # if self.wandb:
        #     wandb.log({"_step": self.global_step})

        print("all internal time:", time.time() - tik)
        return loss

    def validation_step(self, batch, batch_idx):
        self.train()
        print("unpacking!")
        if (batch["geno_tracks"].shape[0] <= 1) or ("ld_window" in batch and batch["ld_window"].sum() == 0):
            return None
        loss, info = self.take_step(batch)

        print(
            "*=" * 50
            + f"\nVal Loss: {loss.item():1.3e} | log_F: {info['mean_log_F']:1.3e} | sig: {torch.exp(self.log_noise).item():1.3e}\n"
            + "*=" * 50
        )
        self.log("val_loss", info["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_err", info["err"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_logdet", info["logdet"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_delta_loss", info["delta_loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_delta_err", info["delta_err"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_delta_logdet", info["delta_logdet"], on_step=False, on_epoch=True, sync_dist=True)
        if "log_F" in batch:
            for key, val in info.items():
                if "error" in key:
                    self.log(f"val_{key}", val, on_step=False, on_epoch=True, sync_dist=True)
        return info

    @rank_zero_only
    def on_validation_epoch_end(
        self,
    ):
        pass

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)

    def configure_optimizers(self):
        base_params = []
        aux_params = []

        for name, param in self.named_parameters():
            if name == "log_noise" or name == "D_weight":
                aux_params.append(param)
            else:
                base_params.append(param)
        param_groups = [
            {"params": base_params, "lr": self.lr},
            {"params": aux_params, "lr": self.lr_sigma},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=int(1e9))

        return {
            "optimizer": optimizer,
            "gradient_clip_val": self.grad_clip_val,
            "weight_decay": self.weight_decay,
            "gradient_clip_algorithm": "norm",
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
