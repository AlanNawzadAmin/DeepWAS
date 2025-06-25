import logging
import multiprocessing
import os
import pickle
import time

import certifi
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from data import get_dataloaders

os.environ["SSL_CERT_FILE"] = certifi.where()

logging.disable(logging.INFO)


class StatsTracker:
    def __init__(self):
        self.geno_mean = torch.tensor(0.0)
        self.geno_stdv = torch.tensor(1.0)
        self.anno_mean = torch.tensor(0.0)
        self.anno_stdv = torch.tensor(1.0)
        self.D_mean = torch.tensor(0.0)
        self.track_stats_idx = 1

    def update_track_stats(self, batch):
        D = batch["D"].float()  # [M]
        geno_tracks = batch["geno_tracks"].float()  # [M, F_1, window_size]
        anno_tracks = batch["anno_tracks"].float()  # [M, F_2]
        n = self.track_stats_idx

        self.D_mean = (n / (n + 1)) * self.D_mean + D.mean() / (n + 1)

        self.geno_mean = (n / (n + 1)) * self.geno_mean + geno_tracks.mean([0, -1]) / (n + 1)
        dev = geno_tracks - self.geno_mean[None, :, None]
        dev = (dev**2).mean([0, -1]) / (n + 1)
        self.geno_stdv = torch.sqrt((n / (n + 1)) * (self.geno_stdv**2) + dev)

        self.anno_mean = (n / (n + 1)) * self.anno_mean + anno_tracks.mean(0) / (n + 1)
        dev = anno_tracks - self.anno_mean[None, :]
        dev = (dev**2).mean(0) / (n + 1)
        self.anno_stdv = torch.sqrt((n / (n + 1)) * (self.anno_stdv**2) + dev)

        self.track_stats_idx += 1


@hydra.main(version_base=None, config_path="../configs", config_name="stats")
def train(cfg: DictConfig) -> None:
    tic = time.time()
    pl.seed_everything(cfg.model.seed, workers=True)
    multiprocessing.set_start_method("spawn")
    out = get_dataloaders(cfg.data)
    train_dataloader, test_dataloader, geno_track_names, anno_track_names = out
    info = {
        "dataset": cfg.data.name,
        "max_n_snps": cfg.data.max_n_snps,
        "geno_tracks": geno_track_names,
        "anno_tracks": anno_track_names,
    }
    st = StatsTracker()

    info["results"] = []
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx > cfg.extra.stop_at:
            break
        t0 = time.time()
        st.update_track_stats(batch)
        t1 = time.time()
        print("*=" * 25 + f"\t{batch_idx:,d}\t" + "*=" * 25)
        print(f"{st.geno_mean.shape=}")
        print(st.geno_mean[:10])
        print(st.geno_stdv[:10])
        print(f"{st.anno_mean.shape=}")
        print(st.anno_mean[:10])
        print(st.anno_stdv[:10])
        print(f"{t1 - t0:.2e} sec")
        print("*=" * 50)
        if (batch_idx % 100 == 0) and (batch_idx > 0):
            save_info(st, info, batch_idx)

    save_info(st, info, "all")
    toc = time.time()
    print(f"{toc - tic:1.3e} sec\n" + "*=" * 50)


def save_info(st, info, idx):
    outfile = f"data/track_stats_{idx}.pkl"
    for key in ["geno_mean", "geno_stdv", "anno_mean", "anno_stdv"]:
        info[key] = getattr(st, key).cpu().numpy()

    print("\n" + "*=" * 50 + f"\nSaving: {outfile=}")
    pickle.dump(info, file=open(outfile, mode="wb"))


if __name__ == "__main__":
    train()
