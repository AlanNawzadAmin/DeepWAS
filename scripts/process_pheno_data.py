import os

import polars as pl

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


def allele_coset(str_):
    a, b = str_.split(":")
    if len(a) == 1 and len(b) == 1:
        return allele_coset_dict[str_]
    else:
        return f"{min(a, b)}:{max(a, b)}"


def process_phenos(chr_):
    # load, create ids, and merge

    loaded_files = []
    genos_dir = "data/ukbb_sumstats/UKBB_409K/"
    for fname in os.listdir(genos_dir):
        print(fname)
        fname = os.path.join(genos_dir, fname)
        chr_stats = pl.read_csv(fname, separator="\t")
        chr_stats = chr_stats.filter(chr_stats["CHR"] == chr_)
        chr_stats = chr_stats.with_columns(
            pl.concat_str([pl.col("A1"), pl.lit(":"), pl.col("A2")])
            .map_elements(
                allele_coset,
                return_dtype=pl.Utf8,
            )
            .alias("allele_coset"),
            (pl.col("A1") < pl.col("A2")).alias("sign"),
        )
        chr_stats = chr_stats.with_columns(
            (pl.col("Beta") / pl.col("se")).alias("Z_score"),
        )
        chr_stats = chr_stats.with_columns([(pl.col("SNP").cast(str) + ":" + pl.col("allele_coset")).alias("var_id")])
        loaded_files.append(chr_stats)
        print(loaded_files[-1].shape)

    suffixes = ["_" + s for s in os.listdir(genos_dir)]
    merged_df = loaded_files[0]
    for df, suffix in zip(loaded_files[1:], suffixes[1:]):
        print(suffix)
        merged_df = merged_df.join(
            df.select(["var_id", "Beta", "EAF", "INFO", "Z_score", "sign"]), on="var_id", how="full", suffix=suffix
        )

    # average across phenotypes

    eaf_cols = [col for col in merged_df.columns if col.startswith("EAF")]

    # Then use them in concat_list
    merged_df = merged_df.with_columns(pl.mean_horizontal(eaf_cols).alias("EAF_mean"))

    info_cols = [col for col in merged_df.columns if col.startswith("INFO")]

    # Then use them in concat_list
    merged_df = merged_df.with_columns(pl.mean_horizontal(info_cols).alias("INFO_mean"))

    id_cols = [col for col in merged_df.columns if col.startswith("var_id")]

    # Then use them in concat_list
    merged_df = merged_df.with_columns(
        pl.concat_list([pl.col(col) for col in id_cols])
        .map_elements(lambda lst: next(x for x in lst if isinstance(x, str)))
        .alias("var_id_mean")
    )

    sign_cols = [col for col in merged_df.columns if col.startswith("sign")]

    # Then use them in concat_list
    merged_df = merged_df.with_columns(
        pl.concat_list([pl.col(col) for col in sign_cols])
        .map_elements(lambda lst: next(x for x in lst if isinstance(x, bool)))
        .alias("sign_mean")
    )

    beta_cols = [col for col in merged_df.columns if col.startswith("Beta")]
    z_cols = [col for col in merged_df.columns if col.startswith("Z_score")]

    # means and formatting
    merged_df = merged_df.select(["var_id_mean", "EAF_mean", "INFO_mean", "sign_mean"] + beta_cols + z_cols)
    merged_df = merged_df.rename(
        {
            "var_id_mean": "var_id",
            "Beta": "Beta_disease_HI_CHOL_SELF_REP.sumstats",
            "Z_score": "Z_score_disease_HI_CHOL_SELF_REP.sumstats",
            "EAF_mean": "EAF",
            "INFO_mean": "INFO",
            "sign_mean": "sign",
        }
    )

    # save
    merge_path = "data/ukbb_sumstats/merged_z_scores"
    fname = os.path.join(merge_path, f"chr{chr_}.arrow")
    merged_df.write_ipc(fname)
    return pl.read_ipc(fname)


for chr_ in range(22):
    process_phenos(chr_ + 1)
