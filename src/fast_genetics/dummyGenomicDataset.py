import torch
from torch.utils.data import Dataset


class DummyGenomicDataset(Dataset):
    def __init__(
        self,
        window_size=100,
        *args,
        **kwargs,
    ):
        self.window_size = window_size
        self.track_names = [f"{idx}" for idx in range(62)]
        self.dbnsfp_col_names = [f"{idx}" for idx in range(6)]
        self.genotype_files = []

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        B = 13
        X = torch.randn(B, 1_006).abs()
        beta = torch.randn(B, 60)
        sign = torch.ones_like(beta[:, 0])
        genomic_tracks = torch.randn(B, len(self.track_names), self.window_size)
        anno_tracks = {f"{idx}": torch.randn(B) for idx in range(len(self.dbnsfp_col_names))}
        return {
            "genotypes": X,
            "region_stats": {"Beta": beta, "EAF": 0.5 * torch.ones(B), "sign": sign},
            "geno_tracks": genomic_tracks,
            "anno_tracks": anno_tracks,
            "position": sign,
        }
