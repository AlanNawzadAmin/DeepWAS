import numpy as np
import torch
from cola.linalg.tbd.logdet_quad import logdet_quad
from torch.linalg import solve_triangular as sotri

from .trainer_base import SNPTrainer


class SNPTrainerRFR(SNPTrainer):
    def __init__(
        self,
        model_class,
        n_pop,
        n_snp,
        use_double=True,
        tol=1e-6,
        max_iters=10_000,
        vtol=1e-2,
        rank_pct=0.10,
        **kwargs,
    ):
        super().__init__(model_class, n_pop, n_snp, **kwargs)
        self.use_double = use_double
        self.tol = tol
        self.max_iters = max_iters
        self.vtol = vtol
        self.rank_pct = rank_pct

    def print_info(self, R, betas, ld_window, diag, log_det, err, sigma, rel_res, n_snp):
        print("*=" * 50 + f"\n{self.__class__.__name__}")
        print(f"{tuple(R.shape)=} | Fraction in LD window: {ld_window.float().mean().item():1.3e}")
        print(f"diag max: {diag.max().item():1.3e} | inv_diag min: {diag.min().item():1.3e}")
        print(f"Solve {rel_res=:1.3e}")
        print(f"{log_det=:1.3e} | {err=:1.3e} | err * sig={err * sigma.item() / n_snp:1.3e}")
        print(f"Beta variance: {((betas[ld_window] ** 2) * self.N).mean().item():1.3e}")
        print("*=" * 50)

    def compare_against_truth(self, batch, log_F_N, info):
        # log_F_tilde = log_F_N - self.log_noise
        log_F_tilde = log_F_N
        return super().compare_against_truth(batch, log_F_tilde, info)

    def regularize_R(self, R):
        return R

    def make_psd(self, R, beta):
        return R, beta

    def basic_predict(self, geno_tracks, anno_tracks, D):  # F_N instead of tilde F
        # return super().basic_predict(geno_tracks, anno_tracks, D) + torch.clamp(self.log_noise, max=1.5)
        return super().basic_predict(geno_tracks, anno_tracks, D)


class WASP(SNPTrainerRFR):
    def forward(self, betas, R, log_F_N, D, ld_window, sigma):
        n_snp = len(log_F_N)
        diag = torch.exp(log_F_N)
        Id = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        # A = (R @ (diag[:, None] * R) + sigma * R + self.eps * Id)[ld_window][:, ld_window]
        A = (R @ (diag[:, None] * R) + self.eps * Id)[ld_window][:, ld_window]
        rhs = (betas * torch.sqrt(self.N))[ld_window][:, None]
        rel_res = 1e-6
        loss, log_det, err = logdet_quad(A, rhs, self.vtol)
        log_det = log_det - n_snp * torch.log(self.N)
        loss = loss - n_snp * torch.log(self.N) + n_snp * np.log(2 * np.pi)
        loss = (0.5 / n_snp) * loss

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + torch.log(self.M)
            F = torch.exp(log_F)

        self.print_info(R, betas, ld_window, diag, log_det, err, sigma, rel_res, n_snp)
        info = {
            "loss": loss,
            "err": err / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info


class WASP_new(SNPTrainerRFR):
    def forward(self, betas, R, log_F_N, D, ld_window, sigma):
        n_snp = len(log_F_N)
        tilde_F = log_F_N - torch.log(sigma)
        diag = torch.exp(-tilde_F)
        ri = R[ld_window][:, ld_window]
        ri.diagonal().add_(self.eps)
        # try:
        # lam, U = torch.linalg.eigh(ri.float())
        lam, U = torch.linalg.eigh(ri)
        lam, U = lam.double(), U.double()
        # except LinAlgError:
        #     lam, U = torch.linalg.eigh(ri)
        lam = lam - self.eps
        lam_inv = torch.where(lam > self.eps, 1 / lam, 0)
        R_inv = U @ (lam_inv[:, None] * U.T)
        R_t = R[ld_window]
        R_ww = R_t.T @ R_inv @ R_t
        A = R_ww / torch.sqrt(diag[:, None])
        A = A / torch.sqrt(diag[None, :])
        A.diagonal().add_(1)

        betas_scaled = betas[ld_window] * torch.sqrt(self.N)
        r_inv_beta = R_inv @ betas_scaled
        rhs = R_t.T @ r_inv_beta
        del R_t

        L = torch.linalg.cholesky(A)
        with torch.no_grad():
            soln = torch.cholesky_solve((rhs / torch.sqrt(diag))[:, None], L)[:, 0] / torch.sqrt(diag)
            rel_res = torch.linalg.norm(torch.sqrt(diag) * (A @ (torch.sqrt(diag) * soln)) - rhs)
            rel_res = rel_res / torch.linalg.norm(rhs)
        err = soln @ (R_ww @ soln) + soln @ (diag * soln)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        bias = (betas_scaled @ r_inv_beta) / sigma + torch.log(torch.where(lam > self.eps, lam * sigma, 1)).sum()

        err = -err / sigma
        err = -err + (2 * err).detach()
        log_det = log_det
        loss = err + log_det
        loss = 0.5 * (loss + (bias - bias.detach()))

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + np.log(11904924)  # torch.log(self.M)
            F = torch.exp(log_F)

        self.print_info(R, betas, ld_window, tilde_F, log_det, err, sigma, rel_res, n_snp)
        info = {
            "n_snp": n_snp,
            "loss": loss,
            "err": err / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info

    def add_delta_loss(self, betas, R, log_F, D, ld_window, sigma, info):
        info["delta_loss"] = 0
        info["delta_err"] = 0
        info["delta_logdet"] = 0


class WASP_alt(SNPTrainerRFR):
    def forward(self, betas, R, log_F_N, D, ld, sigma):
        Ri = R[ld][..., ld]
        Ri = Ri + 1e-4 * torch.eye(Ri.shape[0], dtype=Ri.dtype, device=Ri.device)
        L = torch.linalg.cholesky(Ri)
        RinvRt = sotri(L.T, sotri(L, R[ld], upper=False), upper=True)

        Finv = torch.exp(-log_F_N)
        soln = sotri(L.T, sotri(L, betas[ld][:, None], upper=False), upper=True)[:, 0]
        Up = R[..., ld] @ RinvRt
        A = sigma * (torch.diag(sigma * Finv) + Up)
        rhs = RinvRt.T @ betas[ld] * torch.sqrt(self.N)

        loss, log_det, err = logdet_quad(A, rhs[:, None], self.vtol)

        loss = loss + torch.logdet(Ri) + log_F_N.sum()
        loss = loss + (Ri.shape[0] - 2 * R.shape[0]) * torch.log(sigma)
        loss = loss + (1 / sigma) * betas[ld].T @ soln

        n_snp = len(log_F_N)
        loss = loss - n_snp * torch.log(self.N) + n_snp * np.log(2 * np.pi)
        loss = (0.5 / n_snp) * loss

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + torch.log(self.M)
            F = torch.exp(log_F)

        rel_res = 1e-6
        F_N = torch.exp(log_F_N)
        self.print_info(R, betas, ld, F_N, log_det, err, sigma, rel_res, n_snp)
        info = {
            "loss": loss / n_snp,
            "err": err / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info


class SNPTrainerChol(SNPTrainerRFR):
    def forward(self, betas, R, log_F_N, D, ld_window, sigma):
        n_snp = len(log_F_N)
        diag = torch.exp(log_F_N)
        Id = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        A = (R @ (diag[:, None] * R) + sigma * R + self.eps * Id)[ld_window][:, ld_window]
        rhs = (betas * torch.sqrt(self.N))[ld_window]
        L = torch.linalg.cholesky(A)

        with torch.no_grad():
            soln = sotri(L.T, sotri(L, rhs[:, None], upper=False), upper=True)[:, 0]
            rel_res = torch.linalg.norm(A @ soln - rhs) / torch.linalg.norm(rhs)
            rv = R[:, ld_window] @ soln

        err = rv @ (diag * rv) + sigma * rv[ld_window] @ soln + self.eps * soln @ soln
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        log_det = log_det - n_snp * torch.log(self.N)
        loss = log_det + (2 * err.detach() - err) + n_snp * np.log(2 * np.pi)
        loss = (0.5 / n_snp) * loss.float()

        with torch.no_grad():
            log_F = log_F_N - torch.log(self.N) + torch.log(self.M)
            F = torch.exp(log_F)

        self.print_info(R, betas, ld_window, diag, log_det, err, sigma, rel_res, n_snp)
        info = {
            "loss": loss,
            "err": err / n_snp,
            "logdet": log_det / n_snp,
            "mean_log_F": log_F.mean().item(),
            "mean_F": F.mean().item(),
            "solve_RpFinv": rel_res,
        }
        return loss, info
