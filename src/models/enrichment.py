import torch
from torch.nn import Module


class Enrichment(Module):
    def __init__(
        self,
        window_size,
        enrich_window,
        enrichment_inds,
        enrichment_values,
        enrichment_thresh,
        **kwargs,
    ):
        super().__init__()
        self.wz = window_size
        self.enrich_window = enrich_window
        self.enrichment_inds = enrichment_inds
        self.register_buffer("enrichment_thresh", torch.tensor(enrichment_thresh, dtype=torch.float32))
        self.register_buffer("enrichment_values", torch.tensor(enrichment_values, dtype=torch.float32))

        warn = "f{enrich_window=} is larger than {window_size=}"
        assert enrich_window <= window_size, warn
        warn = "incorrect len on enrichment lists (inds, values, thresh)"
        assert len(enrichment_inds) == len(enrichment_values) == len(enrichment_thresh), warn

    def forward(self, geno_tracks, anno_tracks):
        # geno_tracks: [B,GT,WZ]
        # anno_tracks: [B,GA]
        window = slice(self.wz // 2 - self.enrich_window // 2, self.wz // 2 + self.enrich_window // 2)
        geno_sums = geno_tracks[..., window].sum(-1)
        all_tracks = torch.concat([geno_sums, torch.abs(anno_tracks)], -1)  # [B,GA+GT]
        relevant_tracks = all_tracks[:, self.enrichment_inds]  # [B,E]
        log_F = (relevant_tracks > self.enrichment_thresh).float() @ self.enrichment_values
        return log_F
