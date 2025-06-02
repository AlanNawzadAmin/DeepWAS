import torch


def consistency_check(cfg):
    if cfg.data.use_R_tilde:
        msg = f"R tilde requires include_D=False but got {cfg.model.include_D=}"
        assert not cfg.model.include_D, msg


def get_sqrt(R, eps=1e-4):
    dtype = R.dtype
    R = R.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = torch.maximum(eigvals, torch.tensor(eps))
    R_sqrt = eigvecs @ torch.diag(eigvals**0.5)
    R_sqrt = R_sqrt.to(dtype)
    return R_sqrt


def make_psd(R, thresh=0.0):
    dtype = R.dtype
    R = R.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = torch.maximum(eigvals, torch.tensor(thresh))
    R_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    R_psd = (R_psd + R_psd.T) / 2.0
    R_psd = R_psd.to(dtype)
    return R_psd


def flatten_dict(parent_dict, parent_key="", sep="_"):
    items = []
    for key, val in parent_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(val, dict):
            items.extend(flatten_dict(val, new_key, sep).items())
        else:
            items.append((new_key, val))
    return dict(items)
