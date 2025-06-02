import json
import os

import numpy as np
import polars as pl
import pyBigWig
import torch
from numba import jit
from tqdm import tqdm

allele_coset_dict = {
    "A:T": "A:T",
    "A:C": "A:C",
    "A:G": "A:G",
    "C:T": "A:G",
    "C:A": "A:C",
    "C:G": "C:G",
    "G:T": "A:C",
    "G:C": "C:G",
    "G:A": "A:G",
    "T:A": "A:T",
    "T:C": "A:G",
    "T:G": "A:C",
}

allele_alph_dict = {
    "A:T": "A:T",
    "A:C": "A:C",
    "A:G": "A:G",
    "C:T": "C:T",
    "C:A": "A:C",
    "C:G": "C:G",
    "G:T": "G:T",
    "G:C": "C:G",
    "G:A": "A:G",
    "T:A": "A:T",
    "T:C": "C:T",
    "T:G": "G:T",
}


def allele_coset(str_):
    a, b = str_.split(":")
    if len(a) == 1 and len(b) == 1:
        return allele_coset_dict[str_]
    else:
        return f"{min(a, b)}:{max(a, b)}"


dbnsfp_default_cols = [
    "ESM1b_score",
    "GERP++_RS",
    "SIFT_score",
    "PROVEAN_score",
    "fathmm-XF_coding_score",
    "AlphaMissense_score",
]
rand_fantom_inds = [87, 65, 141, 105, 113, 173, 175, 128, 119, 1, 169, 27, 115, 35, 75, 50, 101, 114, 34, 171]


def load_bigwigs(data_path, first_encode=True, small_fantom=True, include=["fantom", "encode", "phylo"], print_=False):
    all_track_names = []
    dfs = []
    if any(["fantom" in i for i in include]):
        # FANTOM
        fantom_path = data_path / "tracks/fantom"
        experiment_names = os.listdir(fantom_path)
        experiment_names_headers = np.unique([p[:-10] for p in experiment_names])
        if "big_fantom" not in include:
            experiment_names_headers = experiment_names_headers[rand_fantom_inds]
        if print_:
            print("Using headers:", experiment_names_headers)
        experiment_names = [p for p in experiment_names if p[:-10] in experiment_names_headers]
        for p in experiment_names:
            try:
                dfs = dfs + [pyBigWig.open(os.path.join(fantom_path, p))]
                all_track_names = all_track_names + [p]
            except RuntimeError:
                if print_:
                    print("Warning: couldn't load", p)

    # ENCODE
    if any(["encode" in i for i in include]):
        encode_path = data_path / "tracks/encode/"
        experiment_names = [
            p
            for p in os.listdir(encode_path)
            if (os.path.isdir(os.path.join(encode_path, p)) and ("eCLIP" not in p or "all_eCLIP" in p))
        ]
        print(experiment_names)
        if "big_encode" not in include:
            experiment_names = [p for p in experiment_names if "TF_ChIP" not in p]
        for experiment_name in experiment_names:
            if print_:
                print(experiment_name)
            path = os.path.join(encode_path, experiment_name)
            for fname in os.listdir(path):
                if ".metadata.json" in fname:
                    # print(path, fname)
                    json.load(open(os.path.join(path, fname)))
            experiments = [
                json.load(open(os.path.join(path, fname))) for fname in os.listdir(path) if ".metadata.json" in fname
            ]
            # names = [
            #     f"{e['output_type']}_{e['biological_replicates']}_{e['technical_replicates']}" for e in experiments
            # ]
            if first_encode:
                try:
                    # print(experiment_name)
                    chosen_experiment = experiments[0]["accession"]
                    dfs = dfs + [pyBigWig.open(os.path.join(path, f"{chosen_experiment}.bigWig"))]
                    all_track_names = all_track_names + [experiment_name + chosen_experiment]
                except RuntimeError:
                    if print_:
                        print("Warning: couldn't load", experiment_name, chosen_experiment)

    # Phylo
    if "phylo" in include:
        phylo_path = data_path / "tracks/phylo"
        experiment_names = os.listdir(phylo_path)
        for p in experiment_names:
            if print_:
                print(p)
            try:
                dfs = dfs + [pyBigWig.open(os.path.join(phylo_path, p))]
                all_track_names = all_track_names + [p]
            except RuntimeError:
                if print_:
                    print("Warning: couldn't load", p)

    if print_:
        print("N tracks:", len(dfs))
    return dfs, all_track_names


#################################### extract bw ####################################


def get_pos_chunks(positions, chunk_gap=10000):
    chunks = []
    chunk_start = 0
    for i in range(1, len(positions)):
        if positions[i] - positions[chunk_start] > chunk_gap:
            chunks.append((chunk_start, i))
            chunk_start = i
    chunks.append((chunk_start, len(positions)))
    return chunks


def bw_extract_residue_level_values(positions, chrom, dfs, window_size=100, nan_value=0, print_=False):
    pos = positions.numpy()
    sort_idx = np.argsort(pos)
    pos_sorted = pos[sort_idx]
    chunks = get_pos_chunks(pos_sorted)
    results = np.zeros((len(pos_sorted), len(dfs), window_size))
    for chunk_start, chunk_end in tqdm(chunks, position=0, leave=True) if print_ else chunks:
        results[sort_idx[chunk_start:chunk_end]] = bw_extract_chunk(
            pos_sorted[chunk_start:chunk_end], chrom, dfs, window_size, nan_value
        )
    return torch.tensor(results, dtype=torch.float32)


def bw_extract_chunk(pos, chrom, dfs, window_size, nan_value):
    start, end = max(0, pos.min() - window_size // 2), pos.max() + window_size // 2
    results = np.array([df.values(f"chr{chrom}", start, end) for df in dfs])
    windows = np.empty((len(pos), len(dfs), window_size))
    extract_windows(pos, results, windows, window_size, start)
    return np.nan_to_num(windows, nan_value)


@jit(nopython=True)
def extract_windows(pos, all_data, out, window_size, start):
    offset = window_size // 2
    for i, p in enumerate(pos):
        rel_start = p - start - offset
        for j, data in enumerate(all_data):
            out[i, j] = data[rel_start : rel_start + window_size]


#################################### extract pl ####################################


def pl_extract_residue_level_values(
    chr_stats,
    site_var_id,
    col_names=dbnsfp_default_cols,
    id_names=["rs_dbSNP", "pos(1-based)", "ref", "alt"],
    nan_value=0,
    return_sign=False,
):
    """Default is 0 everything. Join by pos:ref:alt accounting for switched ref and alt"""
    snp_positions = pl.DataFrame(
        {"var_id": np.array(site_var_id).astype(str), "position": list(range(len(site_var_id)))}
    )

    if "var_id" not in chr_stats or "sign" not in chr_stats:
        chr_stats = chr_stats.with_columns(
            pl.concat_str([pl.col(id_names[2]), pl.lit(":"), pl.col(id_names[3])])
            .map_elements(
                allele_coset,
                return_dtype=pl.Utf8,
            )
            .alias("allele_coset"),
            (pl.col(id_names[2]) < pl.col(id_names[3])).alias("sign"),
        ).with_columns([(pl.col(id_names[0]).cast(str) + ":" + pl.col("allele_coset")).alias("var_id")])

    result = snp_positions.join(chr_stats, on="var_id", how="left")
    result = result.cast(pl.Float32, strict=False).fill_null(nan_value)
    # aggregate duplciates (from dbnsfp)
    if "Beta" in col_names and "Beta" not in result:
        col_names = [c for c in col_names if ("Beta" not in c and "Z_score" not in c)]
        beta_cols = [c for c in result.columns if "Beta" in c]
        z_cols = [c for c in result.columns if "Z_score" in c]
        result = (
            result.group_by("position")
            .agg([pl.col(col).mean() for col in col_names + beta_cols + z_cols + ["sign"]])
            .sort("position")
        )
        result_dict = {
            name: torch.tensor(result[name].to_list(), dtype=torch.float32)
            for name in col_names + return_sign * ["sign"]
        }
        beta = result.select(beta_cols).to_numpy()
        beta = np.where(beta == 0, np.nan, beta)
        result_dict["Beta"] = torch.tensor(beta, dtype=torch.float32)
        z = result.select(z_cols).to_numpy()
        z = np.where(z == 0, np.nan, z)
        result_dict["z"] = torch.tensor(z, dtype=torch.float32)
    else:
        result = result.group_by("position").agg([pl.col(col).mean() for col in col_names + ["sign"]]).sort("position")
        result_dict = {
            name: torch.tensor(result[name].to_list(), dtype=torch.float32)
            for name in col_names + return_sign * ["sign"]
        }
    return result_dict
