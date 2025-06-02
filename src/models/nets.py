from omegaconf import OmegaConf

from .enformer import Enformer
from .enrichment import Enrichment
from .linear_model import Constant, Linear


def get_model_setup(cfg, geno_track_names, anno_track_names):
    if cfg.model == "linear":
        model_class = Linear
        nn_params = {
            "input_geno_dim": len(geno_track_names),
            "input_anno_dim": len(anno_track_names),
            "window_size": cfg.window_size,
            "average_geno": cfg.average_geno,
            "link": cfg.link,
        }
    elif cfg.model == "constant":
        model_class = Constant
        nn_params = {
            "input_anno_dim": len(anno_track_names),
        }
    elif cfg.model == "enformer":
        model_class = Enformer
        nn_params = {
            "in_dim": len(geno_track_names) + len(anno_track_names),
            **OmegaConf.to_container(cfg, resolve=True),
        }
    elif cfg.model == "enrich":
        model_class = Enrichment
        nn_params = {
            "window_size": cfg.window_size,
            "enrich_window": cfg.enrich_window,
            "enrichment_inds": cfg.enrichment_inds,
            "enrichment_values": cfg.enrichment_values,
            "enrichment_thresh": cfg.enrichment_thresh,
            **OmegaConf.to_container(cfg, resolve=True),
        }
    else:
        raise ValueError(f"{cfg.model=} not found")
    return model_class, nn_params
