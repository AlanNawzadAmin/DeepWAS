import os

import numpy as np
import polars as pl

path = "data/tracks/dbNSFP/dbNSFP5.1a"
for i in np.arange(1, 22 + 1):
    if f"filtered_chr{i}.parquet" not in os.listdir(path):
        print("Processing:", i)
        file = os.path.join(path, f"dbNSFP5.1a_variant.chr{i}")
        df = pl.read_csv(
            file,
            separator="\t",
            null_values=["."],
            infer_schema_length=300000,
            schema_overrides={
                "hg18_chr": pl.Utf8,
                "hg19_chr": pl.Utf8,
                "AllofUs_POPMAX_AC": pl.Utf8,
                "AllofUs_POPMAX_AN": pl.Utf8,
            },
        )
        df_filt = df.filter(df["rs_dbSNP"].is_not_null())
        df_filt.write_parquet(os.path.join(path, f"filtered_chr{i}.parquet"))

# check loading
for i in np.arange(1, 22 + 1):
    print("Processing:", i)
    file = os.path.join(path, f"filtered_chr{i}.parquet")
    df = pl.read_parquet(file)
