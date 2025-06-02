import torch
import torch.nn as nn

link_dict = {
    "exp": nn.Identity(),
    "relu": lambda x: torch.log(torch.clamp(x, min=1e-5)),
    "softplus": lambda x: torch.log(torch.clamp(nn.functional.softplus(x), min=1e-5)),
}


class Linear(nn.Module):
    def __init__(self, input_geno_dim=64, input_anno_dim=64, window_size=64, average_geno=False, link="exp", **kwargs):
        super().__init__()
        self.geno_linear = nn.Linear(input_geno_dim * (window_size if not average_geno else 1), 1)
        self.anno_linear = nn.Linear(input_anno_dim, 1)
        self.average_geno = average_geno
        self.link = link_dict[link]

    def forward(self, geno_tracks, anno_tracks):
        if self.average_geno:
            xg = geno_tracks.mean(-1)
            xg = xg / torch.sqrt(torch.tensor(xg[0].numel()))
        else:
            xg = geno_tracks
            xg = xg / torch.sqrt(torch.tensor(xg[0].numel()))
            xg = xg.reshape(xg.shape[:-2] + (-1,))
        xa = anno_tracks / torch.sqrt(torch.tensor(anno_tracks[0].numel()))
        return self.link(self.geno_linear(xg) + self.anno_linear(xa)).squeeze(-1)


class Constant(nn.Module):
    def __init__(self, input_anno_dim=64, **kwargs):
        super().__init__()
        self.anno_linear = nn.Linear(input_anno_dim, 1)

    def forward(self, geno_tracks, anno_tracks):
        return (self.anno_linear(0 * anno_tracks)).squeeze(-1)
